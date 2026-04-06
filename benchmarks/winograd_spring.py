"""
Winograd evaluation using Solar Spring unified field.
Trains SolarSpringAttention on pronoun resolution data.
Target: 80%+ by learning micro/macro gravity + spring constants.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import sys
sys.path.insert(0, '.')

from benchmarks.winograd_full import WINOGRAD_SCHEMAS
from benchmarks.direct_train import build_generated_pairs, build_vocab
from solar_ring.solar_spring import SolarSpringAttention

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16
D      = 300


def find_pronoun_idx(sentence: str) -> int:
    PRONOUNS = {'it','he','she','they','him','her',
                'them','who','which','that','its'}
    words = sentence.lower().split()
    for i, w in enumerate(words):
        if w.rstrip('.,') in PRONOUNS:
            return i
    return 0


def find_candidate_idx(sentence: str) -> int:
    """Last meaningful word = appended candidate."""
    words = sentence.lower().split()
    for i in range(len(words) - 1, -1, -1):
        w = words[i].rstrip('.,;')
        if len(w) > 1:
            return i
    return len(words) - 1


def _score_pair(out_c, A_c, out_w, A_w,
                sent_c, sent_w, head):
    """
    Score = candidate's backward attention TO pronoun.
    A[candidate_idx, pronoun_idx] — how much the appended
    candidate attends back to the pronoun in ctx.
    """
    L_c = out_c.shape[0]
    L_w = out_w.shape[0]

    p_idx_c = find_pronoun_idx(sent_c)
    p_idx_w = find_pronoun_idx(sent_w)
    cand_c  = find_candidate_idx(sent_c)
    cand_w  = find_candidate_idx(sent_w)

    if (A_c is not None and cand_c < L_c and p_idx_c < L_c):
        attn_score_c = A_c[cand_c, p_idx_c]
        vec_c = out_c[cand_c] + attn_score_c * out_c[p_idx_c]
    else:
        vec_c = out_c.mean(0)

    if (A_w is not None and cand_w < L_w and p_idx_w < L_w):
        attn_score_w = A_w[cand_w, p_idx_w]
        vec_w = out_w[cand_w] + attn_score_w * out_w[p_idx_w]
    else:
        vec_w = out_w.mean(0)

    logit_c = head(vec_c.float())
    logit_w = head(vec_w.float())
    return logit_c, logit_w


def sentence_to_concepts(sentence: str, vocab: dict,
                          depth: int = 0) -> tuple:
    """Convert sentence to concept list and token vectors."""
    words = sentence.lower().split()

    POS_LOOKUP = {
        'he':0,'she':0,'they':0,'him':0,'her':0,
        'it':0,'them':0,'who':0,'which':0,'that':0,
        'the':7,'a':7,'an':7,'this':7,'those':7,
        'is':2,'was':2,'were':2,'are':2,'be':2,
        'not':8,'never':8,'no':8,
    }
    SUBJ_WORDS = {'john','mary','tom','lisa','mike',
                  'anna','bob','sarah','cat','dog',
                  'trophy','ball','window','man','woman'}
    OBJ_WORDS  = {'suitcase','book','car','tree',
                  'table','cup','plate','rock'}

    concepts = []
    vecs     = []
    for t, word in enumerate(words):
        wclean = word.rstrip('.,;')
        if wclean in SUBJ_WORDS:
            pos_idx = 0
        elif wclean in OBJ_WORDS:
            pos_idx = 2
        else:
            pos_idx = POS_LOOKUP.get(wclean, 3)

        concepts.append({
            'pos_idx':   pos_idx,
            'depth':     depth,
            'token_pos': t,
            'slot_idx':  pos_idx,
        })
        wid = vocab.get(wclean, 1)
        vec = torch.zeros(D, device=DEVICE)
        vec[wid % D] = 1.0
        vecs.append(vec)

    vecs_t = (torch.stack(vecs)
              if vecs else torch.zeros(1, D, device=DEVICE))
    return concepts, vecs_t


def train_spring(epochs: int = 30):
    """Train SolarSpringAttention on pronoun resolution."""
    spring = SolarSpringAttention(D).to(DEVICE)
    head   = nn.Linear(D, 1).to(DEVICE)

    optimizer = AdamW(
        list(spring.parameters()) + list(head.parameters()),
        lr=3e-4, weight_decay=0.01
    )

    pairs = build_generated_pairs()

    all_texts = (
        [text for text, _ in pairs] +
        [ctx + ' ' + c for ctx, c, w in WINOGRAD_SCHEMAS] +
        [ctx + ' ' + w for ctx, c, w in WINOGRAD_SCHEMAS]
    )
    vocab = build_vocab(all_texts)

    train_pairs = []
    for i in range(0, len(pairs) - 1, 2):
        text_c, lc = pairs[i]
        text_w, lw = pairs[i + 1]
        if lc == 1 and lw == 0:
            train_pairs.append((text_c, text_w))

    train_pairs = train_pairs[:600]
    print(f"Training on {len(train_pairs)} contrastive pairs, "
          f"{epochs} epochs...")

    for epoch in range(epochs):
        spring.train()
        head.train()
        correct  = 0
        total    = 0
        tot_loss = 0.0

        for sent_c, sent_w in train_pairs:
            optimizer.zero_grad()
            try:
                conc_c, vecs_c = sentence_to_concepts(sent_c, vocab)
                conc_w, vecs_w = sentence_to_concepts(sent_w, vocab)

                out_c, A_c, _ = spring(conc_c, vecs_c)
                out_w, A_w, _ = spring(conc_w, vecs_w)

                logit_c, logit_w = _score_pair(
                    out_c, A_c, out_w, A_w, sent_c, sent_w, head
                )

                # Margin loss only — BCE was hurting
                loss = torch.clamp(
                    1.0 - logit_c.squeeze().float()
                        + logit_w.squeeze().float(),
                    min=0.0
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(spring.parameters()) +
                    list(head.parameters()), 1.0
                )
                optimizer.step()
                tot_loss += loss.item()

                if logit_c.item() > logit_w.item():
                    correct += 1
                total += 1

            except Exception:
                continue

        acc = correct / max(total, 1) * 100
        avg = tot_loss / max(total, 1)
        print(f"Epoch {epoch+1:2d}: "
              f"loss={avg:.4f}  train_acc={acc:.1f}%")

    import os
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(
        {'spring': spring.state_dict(),
         'head':   head.state_dict()},
        'checkpoints/solar_spring.pt'
    )
    print("Saved checkpoints/solar_spring.pt")
    return spring, head, vocab


def evaluate_spring(spring, head, vocab):
    """Evaluate on 90 Winograd schemas."""
    spring.eval()
    head.eval()

    correct = 0
    total   = len(WINOGRAD_SCHEMAS)

    with torch.no_grad():
        for ctx, corr, wrong in WINOGRAD_SCHEMAS:
            try:
                sent_c = ctx + ' ' + corr
                sent_w = ctx + ' ' + wrong

                conc_c, vecs_c = sentence_to_concepts(sent_c, vocab)
                conc_w, vecs_w = sentence_to_concepts(sent_w, vocab)

                out_c, A_c, _ = spring(conc_c, vecs_c)
                out_w, A_w, _ = spring(conc_w, vecs_w)

                lc, lw = _score_pair(
                    out_c, A_c, out_w, A_w, sent_c, sent_w, head
                )

                if lc.item() > lw.item():
                    correct += 1

            except Exception:
                continue

    acc = correct / total * 100
    print(f"\nSolar Spring Winograd: {correct}/{total} = {acc:.1f}%")
    print(f"BERT-base target     : ~70.0%")
    print(f"80% target           : 80.0%")
    print(f"Beats BERT?  {'YES ✓' if acc >= 70 else 'NO'}")
    print(f"Beats 80%?   {'YES ✓' if acc >= 80 else 'NO'}")
    return acc


if __name__ == "__main__":
    print("="*60)
    print("Solar Spring Unified Field — Winograd Benchmark")
    print("="*60)
    print(f"Device: {DEVICE}")

    spring, head, vocab = train_spring(epochs=30)
    acc = evaluate_spring(spring, head, vocab)

    import subprocess
    subprocess.run(['git', 'add',
        'solar_ring/solar_spring.py',
        'benchmarks/winograd_spring.py',
        'checkpoints/solar_spring.pt'])
    subprocess.run(['git', 'commit', '-m',
        f'feat: Solar Spring unified field - Winograd {acc:.1f}%'])
    subprocess.run(['git', 'push', 'origin', 'main'])
    print("Pushed.")
