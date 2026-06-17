"""
Honest Winograd Training Harness
=================================
DO NOT TRAIN without reading the plan section at the bottom.
DO NOT PUSH this file without user approval of the MASS_OPTION.

CHECKPOINT SELECTION RULE (must not drift):
  Checkpoints are selected on TRAIN_ACC computed over the 70 training
  schemas only.  The 20-schema held-out set (results/test_holdout.json)
  and WSC273 novel-267 are read EXACTLY ONCE at the very end — never
  during training, never for early stopping, never for hyperparameter
  choice.  If you find yourself reading test_holdout.json before the
  final evaluation block, you are leaking.

SPLIT PROVENANCE:
  seed=42, Random(42).shuffle(range(90)), train=first 70, test=last 20
  Written to results/test_holdout.json.  Do not regenerate with a new seed.
"""

from __future__ import annotations
import sys, os, json, random, math, re
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Shared model + helpers ────────────────────────────────────────────────────
from benchmarks.winograd_full import WINOGRAD_SCHEMAS
from benchmarks.winograd_80_ls import (
    WinogradSpringModel, get_entity, find_pronoun_idx, PRONOUNS,
)
from solar_ring.solar_spring import SolarSpringAttention
from solar_ring.contextual_embedder import ContextualEmbedder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D      = 384   # MiniLM-L6 output dim

# ─────────────────────────────────────────────────────────────────────────────
# PART A — The fixed train / test split
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_SEED = 42

def make_split():
    """
    Deterministic 70/20 split.  Call once; never alter the seed.
    Returns (train_schemas, test_schemas) as lists of (ctx, correct, wrong).
    The test_schemas MUST NOT be touched until the final evaluation block.
    """
    rng     = random.Random(SPLIT_SEED)
    indices = list(range(90))
    rng.shuffle(indices)
    train_idx = sorted(indices[:70])
    test_idx  = sorted(indices[70:])
    return (
        [WINOGRAD_SCHEMAS[i] for i in train_idx],
        [WINOGRAD_SCHEMAS[i] for i in test_idx],
    )

TRAIN_SCHEMAS, _TEST_SCHEMAS_SEALED = make_split()   # _TEST_SCHEMAS_SEALED not used until eval

# ─────────────────────────────────────────────────────────────────────────────
# PART B — Remove the SUBJ_SET cheat
# ─────────────────────────────────────────────────────────────────────────────
#
# THE CHEAT (benchmarks/winograd_80_ls.py  lines 203-223):
#
#   SUBJ_SET = {
#       'john','mary','tom','trophy','cat','dog',
#       'ball','man','woman','he','she','they',
#       'paul','george','susan','joan','alice',
#       'bob','chris','dave','anna','beth','carol',
#       'sarah','lisa','mike','sam','steve','tim',
#       'emma','diana','rachel','sara','nick','jake',
#       'mark','amy',
#   }
#   pos_idx = 0 if wl in SUBJ_SET else 3   ← binary: either SUBJ(0.95) or PREP(0.20)
#
# The POS_MASS_WEIGHTS table in solar_spring.py maps pos_idx to a weight:
#   pos_idx=0 → 'SUBJ' → 0.95    ← names in SUBJ_SET
#   pos_idx=3 → 'PREP' → 0.20    ← everything else
#
# This 4.75× mass asymmetry dominates all physical forces (gravity, spring,
# decay) and causes the model to attend gravitationally to SUBJ_SET names,
# regardless of what the pronoun actually refers to.  When the training schemas
# happen to have the correct referent be a SUBJ_SET name, the model looks
# accurate.  On novel data it fires on whatever SUBJ_SET name is present.
#
# ── Three replacement options ─────────────────────────────────────────────────
#
# (a) UNIFORM — all content words get the same pos_idx (say 0, same as SUBJ):
#     mass = norm(embed) * 0.95 for every word, regardless of name-list.
#     Honest: fully removes the name-list leakage.  But eliminates all
#     structural differentiation between subjects, objects, and modifiers.
#     The spring now behaves purely as a distance/position model.
#     The G_micro[0,0] term (SUBJ–SUBJ interaction) becomes the only micro
#     gravity learned.  Might be too uniform to discriminate.
#
# (b) REAL_POS — use spaCy to get per-token POS tags, map them to pos_idx:
#     PROPN/NOUN → 0 (SUBJ weight 0.95)
#     VERB       → 1 (0.85)
#     ADJ        → 5 (0.50)
#     ADV        → 6 (0.40)
#     ADP/PREP   → 3 (0.20)
#     DET        → 7 (0.05)
#     other      → 4 (0.15)
#     Honest: removes the name-list leakage.  Proper nouns (PROPN) still get
#     high mass — so both "John" and "Ralph" would get mass 0.95, which is
#     correct (they're both referent candidates).  The cheat was that ONLY
#     some names were in the list.  This requires spaCy at inference time,
#     which adds a dependency.
#
# (c) LEARNED — a tiny MLP (Linear 384→32→1 + Softplus) maps the token
#     embedding to a mass scalar.  Mass becomes a function of the token's
#     meaning rather than a hardcoded list.
#     Honest: completely removes list-based leakage.  The model COULD still
#     learn that embeddings of names-in-training get high mass — but it would
#     have to learn this from the loss signal, not from a hardcoded bias.
#     With only 70 training schemas, risk of a new surface shortcut is real
#     but at least it's learnable from data.  Adds ~12k parameters.
#
# SET THIS BEFORE TRAINING — default is UNIFORM for maximum honesty:
MASS_OPTION = 'UNIFORM'   # 'UNIFORM' | 'REAL_POS' | 'LEARNED'

# ─────────────────────────────────────────────────────────────────────────────
# Model variant with the cheat removed
# ─────────────────────────────────────────────────────────────────────────────

_PRONOUN_WORDS = {
    'it','its','he','him','his','she','her','hers',
    'they','them','their','theirs','who','which','that',
}

class HonestWinogradModel(WinogradSpringModel):
    """
    WinogradSpringModel with build_concepts() de-cheated.
    Mass assignment is controlled by MASS_OPTION.
    Inherits embedder, spring, head from parent.
    """

    def __init__(self, mass_option: str = 'UNIFORM'):
        super().__init__()
        self.mass_option = mass_option

        if mass_option == 'LEARNED':
            # Tiny MLP: embed_dim → 32 → 1, Softplus output
            self.mass_head = nn.Sequential(
                nn.Linear(D, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    # ── OPTION A / B — overrides build_concepts ───────────────────────────
    def build_concepts_honest(self, words: list, spacy_pos: list | None = None) -> list:
        """
        Build concept dicts without consulting any hardcoded name list.

        UNIFORM (a): every non-pronoun token gets pos_idx=0 (same SUBJ weight).
                     Pronouns stay at pos_idx=0 too — uniform across the board.
        REAL_POS (b): pos_idx is mapped from spaCy POS tags.
        """
        SPACY_MAP = {
            'PROPN': 0, 'NOUN': 0,
            'VERB': 1,  'AUX': 1,
            'ADJ':  5,  'ADV': 6,
            'ADP':  3,  'DET': 7,
        }
        concepts = []
        for i, word in enumerate(words):
            if self.mass_option == 'REAL_POS' and spacy_pos is not None:
                tag     = spacy_pos[i] if i < len(spacy_pos) else 'X'
                pos_idx = SPACY_MAP.get(tag, 4)
            else:
                # UNIFORM: everything gets pos_idx=0
                pos_idx = 0
            concepts.append({
                'pos_idx':   pos_idx,
                'depth':     0,
                'token_pos': i,
                'slot_idx':  pos_idx,
            })
        return concepts

    # ── OPTION C — overrides score_from_vecs to inject learned mass ───────
    def score_from_vecs_learned_mass(
        self,
        sentence: str,
        vecs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same as score_from_vecs but injects learned per-token mass into the
        spring before calling forward.  This requires monkey-patching the
        forward call with a custom resonance vector derived from mass_head.

        Implementation note: SolarSpringAttention.forward() accepts a
        sun_resonances list that multiplicatively scales semantic mass via
        (1 + resonance).  We output mass_head(token_embed) as the resonance
        so the spring sees a data-driven mass without any name-list.
        """
        vecs  = vecs.detach().clone()
        words = sentence.lower().split()
        L     = len(words)

        # Compute per-token resonance from embedding (learned mass)
        with torch.no_grad():
            resonances = F.softplus(self.mass_head(vecs.float()))   # (L, 1)
            res_list   = resonances.squeeze(-1).tolist()             # (L,)

        concepts = self.build_concepts_honest(words)

        try:
            out, A, _ = self.spring.forward_with_light_speed(
                concepts, vecs, sun_resonances=res_list,
            )
        except Exception:
            out, A, _ = self.spring(concepts, vecs, sun_resonances=res_list)

        p_idx = min(find_pronoun_idx(words), L - 1)
        c_idx = L - 1

        if A is not None and L > 1:
            attn = A[c_idx, p_idx]
            vec  = out[c_idx] + attn * out[p_idx]
        else:
            vec = out[c_idx]

        pronoun_vec = out[p_idx]
        combined    = torch.cat([vec, pronoun_vec])
        return self.head(combined.float())

    # ── Unified entry point ───────────────────────────────────────────────
    def score_honest(
        self,
        sentence: str,
        vecs: torch.Tensor,
        spacy_pos: list | None = None,
    ) -> torch.Tensor:
        if self.mass_option == 'LEARNED':
            return self.score_from_vecs_learned_mass(sentence, vecs)

        vecs  = vecs.detach().clone()
        words = sentence.lower().split()
        L     = len(words)
        concepts = self.build_concepts_honest(words, spacy_pos)

        try:
            out, A, _ = self.spring.forward_with_light_speed(concepts, vecs)
        except Exception:
            out, A, _ = self.spring(concepts, vecs)

        p_idx = min(find_pronoun_idx(words), L - 1)
        c_idx = L - 1

        if A is not None and L > 1:
            attn = A[c_idx, p_idx]
            vec  = out[c_idx] + attn * out[p_idx]
        else:
            vec = out[c_idx]

        pronoun_vec = out[p_idx]
        combined    = torch.cat([vec, pronoun_vec])
        return self.head(combined.float())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# PART B-continued — ensemble score without the gender leakage
# ─────────────────────────────────────────────────────────────────────────────
#
# The gender_ensemble_score from winograd_95plus.py adds:
#   + 0.5 * log(gender_prior)   ← from hardcoded _MALE_WORDS / _FEMALE_WORDS
#   + 0.3 * predicate_entity_sim
#
# The gender/animacy word lists in solar_memory.py are also a form of
# hardcoded knowledge (they list specific names as male/female).  This is
# LESS of a cheat than SUBJ_SET because gender agreement is real linguistic
# information, not a shortcut for subject detection.  But for clarity, this
# harness uses a simpler scoring path that is honestly separable:

def plain_score(
    model: HonestWinogradModel,
    ctx: str,
    entity: str,
    emb_cache: dict,
) -> float:
    """
    Score = spring output only.  No gender prior.  No predicate sim.
    What the spring + head actually learned, without post-hoc addons.
    """
    sent = ctx + ' ' + entity
    if sent not in emb_cache:
        return 0.0
    vecs = emb_cache[sent].detach().clone()
    with torch.no_grad():
        return model.score_honest(sent, vecs).item()


# ─────────────────────────────────────────────────────────────────────────────
# PART C — Data preparation (training pairs from TRAIN_SCHEMAS only)
# ─────────────────────────────────────────────────────────────────────────────

def build_training_pairs(schemas: list) -> list:
    """
    Build (ctx, entity_correct, entity_wrong, 1) pairs from schemas.
    No augmentation that reaches outside the schema list.
    Winograd completions are in the form "The X did Y." → entity = get_entity(phrase, ctx)
    """
    pairs = []
    for ctx, corr, wrong in schemas:
        ent_c = get_entity(corr,  ctx)
        ent_w = get_entity(wrong, ctx)
        if ent_c != ent_w:
            pairs.append((ctx, ent_c, ent_w, 1))
    return pairs

def precompute_cache(model: HonestWinogradModel, pairs: list) -> dict:
    all_sents = []
    for ctx, ent_c, ent_w, _ in pairs:
        all_sents.append(ctx + ' ' + ent_c)
        all_sents.append(ctx + ' ' + ent_w)
    unique = list(dict.fromkeys(all_sents))
    return model.embedder.embed_words_batch(unique)


# ─────────────────────────────────────────────────────────────────────────────
# PART C — Checkpoint selection rule (TRAIN only)
# ─────────────────────────────────────────────────────────────────────────────

def eval_on_schemas(
    model: HonestWinogradModel,
    schemas: list,
    emb_cache: dict,
    label: str = '',
) -> float:
    """
    Evaluate model on a list of schemas.
    Returns accuracy (float in [0,1]).

    CHECKPOINT SELECTION RULE: call this function with TRAIN_SCHEMAS only
    during training.  Never call it with _TEST_SCHEMAS_SEALED or the
    WSC273 novel set until the final evaluation block at the bottom.
    """
    correct = total = 0
    with torch.no_grad():
        model.spring.eval()
        model.head.eval()
        for ctx, corr, wrong in schemas:
            ent_c = get_entity(corr,  ctx)
            ent_w = get_entity(wrong, ctx)
            if ent_c == ent_w:
                continue
            sc = plain_score(model, ctx, ent_c, emb_cache)
            sw = plain_score(model, ctx, ent_w, emb_cache)
            correct += int(sc > sw)
            total   += 1
    acc = correct / max(total, 1)
    if label:
        print(f"  [{label}] {correct}/{total} = {acc:.1%}")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# PART C — Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model:       HonestWinogradModel,
    train_pairs: list,
    emb_cache:   dict,
    epochs:      int = 60,
    lr:          float = 3e-4,
    ckpt_path:   str = 'checkpoints/honest_winograd_best.pt',
) -> float:
    """
    Train on train_pairs.  Select checkpoint by TRAIN accuracy only.
    Returns best training accuracy seen.
    """
    # ── CHECKPOINT SELECTION RULE — enforced structurally: ───────────────
    # best_ckpt is updated only when train_acc improves.
    # There is no eval on held-out data here.

    params = (list(model.spring.parameters()) +
              list(model.head.parameters()))
    if model.mass_option == 'LEARNED':
        params += list(model.mass_head.parameters())

    optimizer = AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Trainable params: {model.count_parameters():,}")
    print(f"Training pairs  : {len(train_pairs)}")
    print(f"Epochs          : {epochs}  lr={lr}  mass_option={model.mass_option}")

    os.makedirs('checkpoints', exist_ok=True)
    best_train_acc = 0.0

    for epoch in range(epochs):
        model.spring.train()
        model.head.train()
        if model.mass_option == 'LEARNED':
            model.mass_head.train()

        random.shuffle(train_pairs)
        tot_loss = correct = total = 0

        for ctx, ent_c, ent_w, label in train_pairs:
            optimizer.zero_grad()

            sent_c = ctx + ' ' + ent_c
            sent_w = ctx + ' ' + ent_w
            if sent_c not in emb_cache or sent_w not in emb_cache:
                continue

            vecs_c = emb_cache[sent_c].detach().clone()
            vecs_w = emb_cache[sent_w].detach().clone()

            logit_c = model.score_honest(sent_c, vecs_c)
            logit_w = model.score_honest(sent_w, vecs_w)

            # Margin loss (hinge) + BCE
            if label == 1:
                margin = torch.clamp(1.0 - logit_c.squeeze() + logit_w.squeeze(), min=0.0)
                t_c    = torch.ones(1,  device=DEVICE)
                t_w    = torch.zeros(1, device=DEVICE)
            else:
                margin = torch.clamp(1.0 + logit_c.squeeze() - logit_w.squeeze(), min=0.0)
                t_c    = torch.zeros(1, device=DEVICE)
                t_w    = torch.ones(1,  device=DEVICE)

            bce = (F.binary_cross_entropy_with_logits(logit_c.squeeze(), t_c.squeeze()) +
                   F.binary_cross_entropy_with_logits(logit_w.squeeze(), t_w.squeeze()))
            loss = margin + 0.4 * bce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            tot_loss += loss.item()
            if label == 1:
                correct += int(logit_c.item() > logit_w.item())
            else:
                correct += int(logit_c.item() < logit_w.item())
            total += 1

        scheduler.step()
        train_acc = correct / max(total, 1)

        # ── Select checkpoint on TRAIN accuracy — never on held-out ──────
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save({
                'spring':      model.spring.state_dict(),
                'head':        model.head.state_dict(),
                'mass_option': model.mass_option,
                'mass_head':   model.mass_head.state_dict() if model.mass_option == 'LEARNED' else None,
                'train_acc':   train_acc,
                'epoch':       epoch + 1,
                'split_seed':  SPLIT_SEED,
            }, ckpt_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={tot_loss/max(total,1):.4f}  train={train_acc:.1%}  best={best_train_acc:.1%}")

    return best_train_acc


# ─────────────────────────────────────────────────────────────────────────────
# PART D — Final evaluation (run ONCE, at the very end)
# ─────────────────────────────────────────────────────────────────────────────

def final_evaluation(
    model:     HonestWinogradModel,
    ckpt_path: str = 'checkpoints/honest_winograd_best.pt',
):
    """
    CALL THIS EXACTLY ONCE, after training is complete and best checkpoint is loaded.
    Reads test_holdout.json and the WSC273 novel-267 for the FIRST TIME.
    Do not call during training, do not call to tune hyperparameters.
    """
    # Load best checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.spring.load_state_dict(ckpt['spring'])
    model.head.load_state_dict(ckpt['head'])
    if model.mass_option == 'LEARNED' and ckpt.get('mass_head') is not None:
        model.mass_head.load_state_dict(ckpt['mass_head'])
    print(f"\nLoaded checkpoint: epoch={ckpt['epoch']}  train_acc={ckpt['train_acc']:.1%}")
    model.spring.eval()
    model.head.eval()

    # ── Test A: held-out 20 schemas ───────────────────────────────────────
    with open('results/test_holdout.json') as f:
        holdout_data = json.load(f)
    holdout_schemas = [
        (s['context'], s['correct'], s['wrong'])
        for s in holdout_data['schemas']
    ]

    all_holdout_sents = []
    for ctx, corr, wrong in holdout_schemas:
        ent_c = get_entity(corr,  ctx)
        ent_w = get_entity(wrong, ctx)
        all_holdout_sents.extend([ctx + ' ' + ent_c, ctx + ' ' + ent_w])
    holdout_cache = model.embedder.embed_words_batch(list(dict.fromkeys(all_holdout_sents)))

    holdout_acc = eval_on_schemas(model, holdout_schemas, holdout_cache, 'HELD-OUT-20')

    # ── Test B: WSC273 novel-267 ──────────────────────────────────────────
    try:
        import json as _json
        novel_results = []
        with open('results/wsc273_eval.jsonl') as f:
            for line in f:
                r = _json.loads(line)
                if not r['contaminated']:
                    novel_results.append(r)

        # Re-score with the honest model (no gender prior)
        wsc_sents = []
        for r in novel_results:
            wsc_sents.append(r['text'] + ' ' + r['entity_correct'])
            wsc_sents.append(r['text'] + ' ' + r['entity_wrong'])
        wsc_cache = model.embedder.embed_words_batch(list(dict.fromkeys(wsc_sents)))

        wsc_correct = 0
        for r in novel_results:
            sc = plain_score(model, r['text'], r['entity_correct'], wsc_cache)
            sw = plain_score(model, r['text'], r['entity_wrong'],   wsc_cache)
            wsc_correct += int(sc > sw)
        wsc_acc = wsc_correct / max(len(novel_results), 1)
        print(f"  [WSC273-NOVEL-267] {wsc_correct}/{len(novel_results)} = {wsc_acc:.1%}")
    except FileNotFoundError:
        print("  WSC273 results not available — run wsc273_eval.py first")
        wsc_acc = None

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("FINAL HONEST EVALUATION")
    print("=" * 55)
    print(f"  Mass option       : {model.mass_option}")
    print(f"  Best ckpt epoch   : {ckpt['epoch']}")
    print(f"  Train acc (ckpt)  : {ckpt['train_acc']:.1%}")
    print(f"  Held-out-20 acc   : {holdout_acc:.1%}  (vs 50% chance)")
    if wsc_acc is not None:
        print(f"  WSC273-novel-267  : {wsc_acc:.1%}  (vs 50% chance)")
    print("=" * 55)
    return holdout_acc, wsc_acc


# ─────────────────────────────────────────────────────────────────────────────
# PART D — Entry point (do not run without reviewing the plan below)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mass',   default=MASS_OPTION,
                   choices=['UNIFORM', 'REAL_POS', 'LEARNED'],
                   help='Mass assignment option (default: UNIFORM)')
    p.add_argument('--epochs', type=int,   default=60)
    p.add_argument('--lr',     type=float, default=3e-4)
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training; run final_evaluation on existing checkpoint')
    args = p.parse_args()

    model = HonestWinogradModel(mass_option=args.mass).to(DEVICE)
    print(f"HonestWinogradModel | mass_option={args.mass} | params={model.count_parameters():,}")

    ckpt_path = f'checkpoints/honest_winograd_{args.mass.lower()}_best.pt'

    if not args.eval_only:
        train_pairs = build_training_pairs(TRAIN_SCHEMAS)
        emb_cache   = precompute_cache(model, train_pairs)

        # Baseline before training
        print("\nBaseline (untrained) on training schemas:")
        eval_on_schemas(model, TRAIN_SCHEMAS, emb_cache, 'TRAIN-BASELINE')

        print("\nTraining ...")
        best = train(model, train_pairs, emb_cache,
                     epochs=args.epochs, lr=args.lr, ckpt_path=ckpt_path)
        print(f"\nTraining done.  Best train acc: {best:.1%}")

    print("\nFinal evaluation (test_holdout + WSC273 novel):")
    final_evaluation(model, ckpt_path)
