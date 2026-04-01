"""Direct pronoun-resolution training — binary classification approach.

Instead of next-token prediction, each model scores correct pronoun
resolutions higher than wrong ones (BCELoss).

Dataset:
  Training : 70 Winograd schemas × 2  +  1000 generated items  =  1140 items
  Eval     : 20 held-out schemas  (pairwise, per-epoch accuracy)
  Final    : all 90 Winograd schemas  (pairwise comparison table)

Models:
  SolarClassifier   — SolarRingModel  + pronoun_head(d_model → 1)
  BiLSTMClassifier  — BiLSTM          + pronoun_head(d_model → 1)
  LSTMClassifier    — VanillaLSTM     + pronoun_head(d_model → 1)

Context vector:
  Solar Ring  : mem_vec  (flattened ring memory projected to d_model)
  BiLSTM      : cat(h_n[-2], h_n[-1])  (last bidirectional hidden, d_model)
  VanillaLSTM : h_n[-1]                (last layer hidden, d_model)

Usage:
  python benchmarks/direct_train.py
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple

# ── Device / CUDA ─────────────────────────────────────────────────────────────
print(f"Torch : {torch.__version__}")
print(f"CUDA  : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

from solar_ring.model        import SolarRingModel
from solar_ring.config       import D_MODEL, GRAD_CLIP
from solar_ring.glove_loader import load_glove
from baseline.vanilla_lstm   import VanillaLSTM
from baseline.bilstm         import BiLSTM
from benchmarks.winograd_full import (
    WINOGRAD_SCHEMAS,
    _normalize,
    _word_tokenize,
    _pronoun_category,
)

EPOCHS        = 30
LR            = 1e-3
SOLAR_DROPOUT = 0.35   # slightly higher than baselines to curb Solar Ring overfit
DROPOUT       = 0.3
EARLY_STOP    = 5      # patience: stop if eval doesn't improve for this many epochs
GLOVE_PATH    = "data/glove.6B.300d.txt"
EVAL_SPLIT    = 20     # Winograd schemas held out for per-epoch accuracy


# ── Classifier wrappers ───────────────────────────────────────────────────────

class SolarClassifier(nn.Module):
    """SolarRingModel backbone + binary pronoun_head.

    Context vector: mem_vec returned in aux["context_vec"] — the flattened
    ring-memory projected to d_model.  This vector encodes the full
    subject/object pole state after reading the entire input.
    """

    def __init__(self, vocab_size: int, pretrained_embeddings=None):
        super().__init__()
        self.base         = SolarRingModel(vocab_size, pretrained_embeddings)
        self.drop         = nn.Dropout(SOLAR_DROPOUT)   # 0.35 — slightly higher
        self.pronoun_head = nn.Linear(D_MODEL, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids : (T,)  →  score : scalar in [0, 1]"""
        _, aux = self.base(token_ids)
        ctx    = aux["context_vec"]                              # (D_MODEL,)
        return torch.sigmoid(
            self.pronoun_head(self.drop(ctx.float()))
        ).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BiLSTMClassifier(nn.Module):
    """BiLSTM backbone + binary pronoun_head.

    Context vector: cat(h_n[-2], h_n[-1]) — forward and backward hidden
    states of the last BiLSTM layer concatenated to d_model.
    """

    def __init__(self, vocab_size: int, pretrained_embeddings=None):
        super().__init__()
        self.base         = BiLSTM(vocab_size, pretrained_embeddings)
        self.drop         = nn.Dropout(DROPOUT)
        self.pronoun_head = nn.Linear(D_MODEL, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, hidden = self.base(token_ids.unsqueeze(0))
        h_n = hidden[0]   # (num_layers*2, 1, D_MODEL//2) = (4, 1, 150)
        ctx = torch.cat([h_n[-2, 0, :], h_n[-1, 0, :]], dim=-1).float()  # (D_MODEL,)
        return torch.sigmoid(
            self.pronoun_head(self.drop(ctx))
        ).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LSTMClassifier(nn.Module):
    """VanillaLSTM backbone + binary pronoun_head.

    Context vector: h_n[-1] — last layer hidden state at the final token.
    """

    def __init__(self, vocab_size: int, pretrained_embeddings=None):
        super().__init__()
        self.base         = VanillaLSTM(vocab_size, pretrained_embeddings)
        self.drop         = nn.Dropout(DROPOUT)
        self.pronoun_head = nn.Linear(D_MODEL, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, hidden = self.base(token_ids.unsqueeze(0))
        ctx = hidden[0][-1, 0, :].float()   # (D_MODEL,)
        return torch.sigmoid(
            self.pronoun_head(self.drop(ctx))
        ).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Generated pronoun pairs ───────────────────────────────────────────────────

def build_generated_pairs() -> List[Tuple[str, int]]:
    """
    500 pronoun-resolution pairs (1000 items: correct + wrong each).

    IT  — 200 pairs  (4 patterns × 50 variations)
    HE  — 150 pairs  (3 patterns × 50 variations)
    SHE — 150 pairs  (3 patterns × 50 variations)

    Returns flat list of (text, label) where text = context + " " + completion.
    """
    IT     = ["trophy","suitcase","ball","window","cup",
              "plate","box","vase","rock","bottle"]
    MALE   = ["John","Paul","Tom","Mike","Sam",
              "Bob","Steve","Jake","Chris","Alex"]
    FEMALE = ["Mary","Anna","Lisa","Sarah","Beth",
              "Emma","Diana","Carol","Rachel","Nina"]

    SYN_FRAGILE = ["fragile","delicate","brittle","breakable","flimsy",
                   "frail","weak","thin","crumbly","soft"]
    SYN_SMALL   = ["small","tiny","compact","narrow","little",
                   "slim","short","petite","miniature","light"]
    SYN_SHARP   = ["sharp","pointed","jagged","edged","spiky",
                   "thorny","barbed","dangerous","piercing","cutting"]
    SYN_CRACK   = ["cracked","dented","chipped","scratched","damaged",
                   "bent","broke","split","fractured","shattered"]
    SYN_LEAVE   = ["leave","go","stop","quit","move",
                   "return","resign","rest","wait","depart"]
    SYN_SKILLED = ["skilled","talented","qualified","competent","experienced",
                   "capable","gifted","diligent","proficient","expert"]
    SYN_NEED    = ["help","assistance","support","advice","guidance",
                   "information","directions","funds","backup","resources"]
    SYN_REST    = ["rest","relax","sleep","recover","pause",
                   "sit","breathe","calm","nap","unwind"]
    SYN_TIRED   = ["tired","exhausted","sick","hurt","stressed",
                   "upset","weak","unwell","injured","dizzy"]
    SYN_ADVICE  = ["advice","help","support","guidance","assistance",
                   "opinions","feedback","suggestions","directions","insights"]

    N = lambda lst, i, off=0: lst[(i + off) % len(lst)]
    items: List[Tuple[str, int]] = []

    # ── IT: 4 patterns × 50 = 200 pairs → 400 items ──────────────────────────

    # IT-A: "broke because fragile" → it = n2
    for i in range(50):
        n1, n2, kw = N(IT,i), N(IT,i,3), N(SYN_FRAGILE, i // 5)
        ctx = f"The {n1} broke the {n2} because it was {kw}."
        items.append((ctx + f" The {n2} was {kw}.", 1))
        items.append((ctx + f" The {n1} was {kw}.", 0))

    # IT-B: "fit in because small" → it = n1
    for i in range(50):
        n1, n2, kw = N(IT,i), N(IT,i,3), N(SYN_SMALL, i // 5)
        ctx = f"The {n1} fit in the {n2} because it was {kw}."
        items.append((ctx + f" The {n1} was {kw}.", 1))
        items.append((ctx + f" The {n2} was {kw}.", 0))

    # IT-C: "avoided because sharp" → it = n2
    for i in range(50):
        n1, n2, kw = N(IT,i), N(IT,i,3), N(SYN_SHARP, i // 5)
        ctx = f"The {n1} avoided the {n2} because it was {kw}."
        items.append((ctx + f" The {n2} was {kw}.", 1))
        items.append((ctx + f" The {n1} was {kw}.", 0))

    # IT-D: "fell and cracked" → it = n2
    for i in range(50):
        n1, n2, kw = N(IT,i), N(IT,i,3), N(SYN_CRACK, i // 5)
        ctx = f"The {n1} fell on the {n2} and it {kw}."
        items.append((ctx + f" The {n2} {kw}.", 1))
        items.append((ctx + f" The {n1} {kw}.", 0))

    assert len(items) == 400, f"IT: expected 400 items, got {len(items)}"

    # ── HE: 3 patterns × 50 = 150 pairs → 300 items ──────────────────────────

    # HE-A: "told he should leave" → he = n2
    for i in range(50):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_LEAVE, i // 5)
        ctx = f"{n1} told {n2} that he should {kw}."
        items.append((ctx + f" {n2} should {kw}.", 1))
        items.append((ctx + f" {n1} should {kw}.", 0))

    # HE-B: "hired because skilled" → he = n2
    for i in range(50):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_SKILLED, i // 5)
        ctx = f"{n1} hired {n2} because he was {kw}."
        items.append((ctx + f" {n2} was {kw}.", 1))
        items.append((ctx + f" {n1} was {kw}.", 0))

    # HE-C: "called because needed" → he = n1
    for i in range(50):
        n1, n2, kw = N(MALE,i), N(MALE,i,3), N(SYN_NEED, i // 5)
        ctx = f"{n1} called {n2} because he needed {kw}."
        items.append((ctx + f" {n1} needed {kw}.", 1))
        items.append((ctx + f" {n2} needed {kw}.", 0))

    # ── SHE: 3 patterns × 50 = 150 pairs → 300 items ─────────────────────────

    # SHE-A: "told she should rest" → she = n2
    for i in range(50):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_REST, i // 5)
        ctx = f"{n1} told {n2} that she should {kw}."
        items.append((ctx + f" {n2} should {kw}.", 1))
        items.append((ctx + f" {n1} should {kw}.", 0))

    # SHE-B: "helped because tired" → she = n2
    for i in range(50):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_TIRED, i // 5)
        ctx = f"{n1} helped {n2} because she was {kw}."
        items.append((ctx + f" {n2} was {kw}.", 1))
        items.append((ctx + f" {n1} was {kw}.", 0))

    # SHE-C: "called because needed advice" → she = n1
    for i in range(50):
        n1, n2, kw = N(FEMALE,i), N(FEMALE,i,3), N(SYN_ADVICE, i // 5)
        ctx = f"{n1} called {n2} because she needed {kw}."
        items.append((ctx + f" {n1} needed {kw}.", 1))
        items.append((ctx + f" {n2} needed {kw}.", 0))

    assert len(items) == 1000, f"Expected 1000 items (500 pairs), got {len(items)}"

    # ── HARD patterns: 5 templates × 60 = 300 pairs → 600 items ─────────────
    # These require semantic reasoning, not just syntactic proximity.

    FEMALE2  = ["Sarah","Emily","Rachel","Diana","Carol",
                "Anna","Lisa","Beth","Emma","Nina"]
    MALE2    = ["the manager","the employee","the coach","the student","the doctor",
                "the teacher","the lawyer","the client","the director","the analyst"]
    NOUNS2   = ["scientist","theory","dog","man","judge",
                "witness","author","claim","athlete","pilot"]

    SYN_TRAIN  = ["trained","practiced","prepared","worked","studied",
                  "drilled","rehearsed","exercised","competed","improved"]
    SYN_PROM   = ["promoted","hired","selected","chosen","recommended",
                  "praised","rewarded","advanced","appointed","recognized"]
    SYN_CONTROV= ["controversial","disputed","challenged","debated","questioned",
                  "contested","unpopular","radical","unproven","complex"]
    SYN_WRONG  = ["wrong","mistaken","incorrect","confused","misled",
                  "inaccurate","off-track","uninformed","biased","naive"]
    SYN_PROV   = ["provoked","threatened","startled","cornered","attacked",
                  "scared","disturbed","bothered","aggravated","alarmed"]

    # HARD-1: "X knew Y would win because she had trained harder." → she = Y
    for i in range(60):
        n1, n2, kw = N(FEMALE2,i), N(FEMALE2,i,3), N(SYN_TRAIN, i // 6)
        ctx = f"{n1} knew that {n2} would win because she had {kw} harder."
        items.append((ctx + f" {n2} had {kw} harder.", 1))
        items.append((ctx + f" {n1} had {kw} harder.", 0))

    # HARD-2: "The manager told the employee that he was promoted." → he = employee
    for i in range(60):
        n1, n2, kw = N(MALE2,i), N(MALE2,i,3), N(SYN_PROM, i // 6)
        ctx = f"The {n1} told the {n2} that he was {kw}."
        items.append((ctx + f" The {n2} was {kw}.", 1))
        items.append((ctx + f" The {n1} was {kw}.", 0))

    # HARD-3: "The scientist proved the theory although it was controversial." → it = theory
    for i in range(60):
        n1, n2, kw = N(NOUNS2,i), N(NOUNS2,i,3), N(SYN_CONTROV, i // 6)
        ctx = f"The {n1} proved the {n2} although it was {kw}."
        items.append((ctx + f" The {n2} was {kw}.", 1))
        items.append((ctx + f" The {n1} was {kw}.", 0))

    # HARD-4: "Anna believed Lisa was wrong because she had seen the evidence." → she = Anna
    for i in range(60):
        n1, n2, kw = N(FEMALE2,i), N(FEMALE2,i,3), N(SYN_WRONG, i // 6)
        ctx = f"{n1} believed {n2} was {kw} because she had seen the evidence."
        items.append((ctx + f" {n1} had seen the evidence.", 1))
        items.append((ctx + f" {n2} had seen the evidence.", 0))

    # HARD-5: "The dog bit the man because it was provoked." → it = dog
    for i in range(60):
        n1, n2, kw = N(NOUNS2,i), N(NOUNS2,i,3), N(SYN_PROV, i // 6)
        ctx = f"The {n1} confronted the {n2} because it was {kw}."
        items.append((ctx + f" The {n1} was {kw}.", 1))
        items.append((ctx + f" The {n2} was {kw}.", 0))

    assert len(items) == 1600, f"Expected 1600 items (800 pairs), got {len(items)}"
    return items


# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(texts: List[str], max_vocab: int = 5000) -> Dict[str, int]:
    word2id: Dict[str, int] = {"<UNK>": 0, "<PAD>": 1}
    for text in texts:
        for w in text.split():
            c = _normalize(w)
            if c and c not in word2id and len(word2id) < max_vocab:
                word2id[c] = len(word2id)
    return word2id


def encode(text: str, word2id: Dict[str, int], max_len: int = 128) -> torch.Tensor:
    unk = word2id.get("<UNK>", 0)
    ids = [word2id.get(_normalize(w), unk) for w in text.split() if _normalize(w)]
    return torch.tensor(ids[:max_len], dtype=torch.long)


# ── Per-epoch evaluation (pairwise on held-out schemas) ───────────────────────

def eval_schemas(model: nn.Module, schemas, word2id: Dict[str, int]) -> float:
    model.eval()
    correct = 0
    with torch.no_grad(), torch.autocast(DEVICE.type, dtype=DTYPE):
        for ctx, corr, wrong in schemas:
            ids_c = encode(ctx + " " + corr,  word2id).to(DEVICE)
            ids_w = encode(ctx + " " + wrong, word2id).to(DEVICE)
            if ids_c.numel() == 0 or ids_w.numel() == 0:
                continue
            sc = model(ids_c).item()
            sw = model(ids_w).item()
            if sc > sw:
                correct += 1
    return correct / max(len(schemas), 1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_classifier(
    model:        nn.Module,
    model_label:  str,
    train_items:  List[Tuple[str, int]],
    eval_schemas_list,
    word2id:      Dict[str, int],
    ckpt_name:    str,
) -> nn.Module:
    print(f"\n{'='*62}\nTraining {model_label}\n{'='*62}")
    print(f"  params={model.count_parameters():,}  "
          f"train={len(train_items)} items  eval={len(eval_schemas_list)} schemas")

    # Pre-encode all training items
    encoded = []
    for text, label in train_items:
        ids = encode(text, word2id)
        if ids.numel() > 0:
            encoded.append((ids, torch.tensor(float(label), dtype=torch.float32)))

    opt       = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    # epoch 1-10: lr=1e-3, epoch 11-20: lr=3e-4, epoch 21-30: lr=1e-4
    sched     = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)
    criterion = nn.BCELoss()
    ckpt_path = Path(f"checkpoints/{ckpt_name}")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    patience = 0
    indices  = list(range(len(encoded)))

    for ep in range(EPOCHS):
        model.train()
        random.shuffle(indices)
        ep_loss = 0.0

        for idx in indices:
            ids, label = encoded[idx]
            ids   = ids.to(DEVICE)
            label = label.to(DEVICE)

            opt.zero_grad()
            with torch.autocast(DEVICE.type, dtype=DTYPE):
                score = model(ids)              # scalar
            loss = criterion(score.float(), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ep_loss += loss.item()

        acc      = eval_schemas(model, eval_schemas_list, word2id)
        avg_loss = ep_loss / max(len(encoded), 1)
        lr_now   = opt.param_groups[0]["lr"]
        sched.step()

        marker = ""
        if acc > best_acc:
            best_acc = acc
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = "  ✓ best"
        else:
            patience += 1
            marker = f"  (patience {patience}/{EARLY_STOP})"

        print(f"  Epoch {ep+1:02d}/{EPOCHS}  loss={avg_loss:.4f}  eval={acc:.1%}"
              f"  lr={lr_now:.0e}{marker}")

        if patience >= EARLY_STOP:
            print(f"  Early stopping at epoch {ep+1}")
            break

    print(f"  Loading best checkpoint (eval={best_acc:.1%})")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model


# ── Final evaluation (all 90 schemas) ────────────────────────────────────────

def evaluate_classifier(
    model:      nn.Module,
    model_name: str,
    word2id:    Dict[str, int],
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    cat_correct: Dict[str, int] = {}
    cat_total:   Dict[str, int] = {}

    with torch.no_grad(), torch.autocast(DEVICE.type, dtype=DTYPE):
        for ctx, corr, wrong in WINOGRAD_SCHEMAS:
            cat = _pronoun_category(ctx)
            cat_total[cat]   = cat_total.get(cat, 0) + 1
            cat_correct[cat] = cat_correct.get(cat, 0)

            ids_c = encode(ctx + " " + corr,  word2id).to(DEVICE)
            ids_w = encode(ctx + " " + wrong, word2id).to(DEVICE)
            sc = model(ids_c).item() if ids_c.numel() > 0 else 0.0
            sw = model(ids_w).item() if ids_w.numel() > 0 else 0.0
            if sc > sw:
                cat_correct[cat] += 1

    total         = len(WINOGRAD_SCHEMAS)
    total_correct = sum(cat_correct.values())
    acc           = total_correct / total

    print(f"\n  {model_name}: {total_correct}/{total} = {acc:.1%}")
    for cat in ["IT", "HE", "SHE"]:
        if cat in cat_total:
            c, t = cat_correct.get(cat, 0), cat_total[cat]
            print(f"    {cat}  pronouns: {c}/{t} = {c/t:.1%}")

    cat_accs = {cat: cat_correct.get(cat, 0) / cat_total[cat] for cat in cat_total}
    return acc, cat_accs


# ── Table helpers ─────────────────────────────────────────────────────────────

def _pct(v):   return f"{v:.1%}" if v is not None else "n/a"
def _delta(d): return f"+{d:.1%}" if d >= 0 else f"{d:.1%}"

def _print_table(rows, headers):
    col_w = [max(len(h), max(len(str(r[i])) for r in rows))
             for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in col_w)
    fmt = " | ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("Solar Ring — Direct Pronoun-Resolution Training")
    print("Binary classification  |  BCELoss  |  context-vector scoring")
    print("=" * 62)

    # Shuffle and split Winograd schemas
    random.seed(42)
    schemas = list(WINOGRAD_SCHEMAS)
    random.shuffle(schemas)
    train_schemas = schemas[:70]   # 70 schemas → 140 (ctx+correct / ctx+wrong) items
    eval_schemas  = schemas[70:]   # 20 held-out schemas for per-epoch eval

    # Build training pairs
    wino_items: List[Tuple[str, int]] = []
    for ctx, corr, wrong in train_schemas:
        wino_items.append((ctx + " " + corr,  1))
        wino_items.append((ctx + " " + wrong, 0))

    gen_items   = build_generated_pairs()           # 1000 items (500 pairs)
    all_items   = wino_items + gen_items             # 140 + 1000 = 1140 items

    print(f"\n[0] Dataset")
    print(f"    Winograd train items : {len(wino_items)} (70 schemas × 2)")
    print(f"    Generated items      : {len(gen_items)} (500 base + 300 hard pairs × 2)")
    print(f"    Total training items : {len(all_items)}")
    print(f"    Eval schemas         : {len(eval_schemas)} (held-out Winograd)")

    # Vocabulary from all training text + all Winograd text (for eval coverage)
    all_texts  = [text for text, _ in all_items]
    wino_texts = [t for ctx, c, w in WINOGRAD_SCHEMAS for t in (ctx, c, w)]
    word2id    = build_vocab(all_texts + wino_texts, max_vocab=5000)
    print(f"    Vocab                : {len(word2id)} tokens")

    # GloVe embeddings
    print(f"\n[1] Loading GloVe 300d from {GLOVE_PATH}...")
    glove = None
    if Path(GLOVE_PATH).exists():
        glove = load_glove(GLOVE_PATH, word2id, d=300)
        print(f"    Matrix : {glove.shape}")
    else:
        print(f"    WARNING: {GLOVE_PATH} not found — random init")

    vs = len(word2id)

    # Instantiate classifiers
    solar_clf  = SolarClassifier(vs,  glove).to(DEVICE, DTYPE)
    bilstm_clf = BiLSTMClassifier(vs, glove).to(DEVICE, DTYPE)
    lstm_clf   = LSTMClassifier(vs,   glove).to(DEVICE, DTYPE)

    # Untrained baselines
    print("\n[2] Untrained baselines on all 90 Winograd schemas...")
    solar_before,  solar_cats_before  = evaluate_classifier(
        solar_clf,  "SolarRing   (untrained)", word2id)
    bilstm_before, bilstm_cats_before = evaluate_classifier(
        bilstm_clf, "BiLSTM      (untrained)", word2id)
    lstm_before,   lstm_cats_before   = evaluate_classifier(
        lstm_clf,   "VanillaLSTM (untrained)", word2id)

    # Train
    print("\n[3] Training SolarClassifier  (20 epochs, BCELoss)...")
    solar_clf  = train_classifier(
        solar_clf,  "SolarClassifier",
        all_items, eval_schemas, word2id, "solar_direct_best.pt")

    print("\n[4] Training BiLSTMClassifier (20 epochs, BCELoss)...")
    bilstm_clf = train_classifier(
        bilstm_clf, "BiLSTMClassifier",
        all_items, eval_schemas, word2id, "bilstm_direct_best.pt")

    print("\n[5] Training LSTMClassifier   (20 epochs, BCELoss)...")
    lstm_clf   = train_classifier(
        lstm_clf,   "LSTMClassifier",
        all_items, eval_schemas, word2id, "lstm_direct_best.pt")

    # Final evaluation
    print("\n[6] Final evaluation on all 90 Winograd schemas (trained)...")
    solar_after,  solar_cats  = evaluate_classifier(
        solar_clf,  "SolarRing   (trained)", word2id)
    bilstm_after, bilstm_cats = evaluate_classifier(
        bilstm_clf, "BiLSTM      (trained)", word2id)
    lstm_after,   lstm_cats   = evaluate_classifier(
        lstm_clf,   "VanillaLSTM (trained)", word2id)

    # Results table
    print("\n" + "=" * 62)
    print(f"Results — 90 Winograd Schemas  ({EPOCHS} epochs, direct classification)")
    print("=" * 62)

    def row(name, before, after, cats, params):
        return [
            name,
            _pct(before),
            _pct(after),
            _delta(after - before),
            _pct(cats.get("IT")),
            _pct(cats.get("HE")),
            _pct(cats.get("SHE")),
            f"{params:,}",
        ]

    headers = ["Model", "Untrained", f"After {EPOCHS}ep", "Gain",
               "IT%", "HE%", "SHE%", "Params"]
    rows = [
        row("Solar Ring",   solar_before,  solar_after,  solar_cats,
            solar_clf.count_parameters()),
        row("BiLSTM",       bilstm_before, bilstm_after, bilstm_cats,
            bilstm_clf.count_parameters()),
        row("Vanilla LSTM", lstm_before,   lstm_after,   lstm_cats,
            lstm_clf.count_parameters()),
    ]
    print()
    _print_table(rows, headers)

    solar_he  = solar_cats.get("HE",  0);  bilstm_he  = bilstm_cats.get("HE",  0)
    solar_she = solar_cats.get("SHE", 0);  bilstm_she = bilstm_cats.get("SHE", 0)
    solar_it  = solar_cats.get("IT",  0)

    print(f"\nTarget: Solar Ring > 68%       → "
          f"{'MET ✓' if solar_after > 0.68 else f'gap {0.68 - solar_after:.1%}'}")
    print(f"Solar Ring beats BiLSTM?      → "
          f"{'YES ✓' if solar_after > bilstm_after else 'NO'}")
    print(f"Solar Ring HE  > BiLSTM HE?   → "
          f"{'YES ✓' if solar_he  > bilstm_he  else 'NO'}"
          f"  ({_pct(solar_he)} vs {_pct(bilstm_he)})")
    print(f"Solar Ring SHE > BiLSTM SHE?  → "
          f"{'YES ✓' if solar_she > bilstm_she else 'NO'}"
          f"  ({_pct(solar_she)} vs {_pct(bilstm_she)})")
    print(f"Solar Ring IT  > 65%?         → "
          f"{'YES ✓' if solar_it > 0.65 else f'gap {0.65 - solar_it:.1%}'}"
          f"  ({_pct(solar_it)})")


if __name__ == "__main__":
    main()
