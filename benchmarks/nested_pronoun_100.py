"""NestedPronoun-100 benchmark.

100 sentences across 5 groups probing pronoun resolution at increasing depth.

Group 1 — depth 2  (planet level,   20 pairs): embedded clause, "it" → subject or object
Group 2 — depth 3  (moon level,     20 pairs): doubly-embedded rel-clause + "it"
Group 3 — depth 4  (sub-moon,       20 pairs): triply-embedded clause, deep subject ID
Group 4 — cross    (cross-depth,    20 pairs): pronoun at moon level → sun subject
Group 5 — multi    (multi-pronoun,  20 pairs): two pronouns in one sentence, both resolved

Evaluation:
  For Groups 1-4: score(sentence + correct) > score(sentence + wrong) → correct
  For Group 5   : both (pronoun1, pronoun2) pairs must resolve correctly

Solar Ring advantage: orbital walk naturally follows depth hierarchy.
LSTM expected to degrade sharply from depth 2 → depth 4.

Usage:
  python benchmarks/nested_pronoun_100.py
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Union

# ── Device ─────────────────────────────────────────────────────────────────────
print(f"Torch : {torch.__version__}")
print(f"CUDA  : {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

from solar_ring.config       import D_MODEL
from solar_ring.glove_loader import load_glove
from benchmarks.direct_train import (
    SolarClassifier, BiLSTMClassifier, LSTMClassifier,
    build_vocab,
)
from benchmarks.winograd_full import WINOGRAD_SCHEMAS
from benchmarks.structured_qa import rebuild_direct_vocab, _normalize, encode

GLOVE_PATH = "data/glove.6B.300d.txt"


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# Each standard item: (sentence, pronoun_label, correct_word, wrong_word, depth)
# Group 5 item:       (sentence, [(p1,c1,w1),(p2,c2,w2)], "multi")
# ══════════════════════════════════════════════════════════════════════════════

# ── Group 1: depth 2 — planet-ring, simple embedded clause ────────────────────
# "it" follows the embedded clause; context determines referent.
# correct = the noun "it" refers to;  wrong = the other noun in the clause.
GROUP1: List[Tuple[str, str, str, str, int]] = [
    # ─ Given (5) ─
    ("John told Mary that the cat chased the dog. It was angry.",     "it", "cat",    "dog",    2),
    ("Sarah said that the ball broke the window. It shattered.",       "it", "window", "ball",   2),
    ("Mike told Lisa that the dog bit the man. It was angry.",         "it", "dog",    "man",    2),
    ("Anna said that the car hit the tree. It fell.",                  "it", "tree",   "car",    2),
    ("Tom told Beth that the cat ate the fish. It was hungry.",        "it", "cat",    "fish",   2),
    # ─ Generated (15) ─
    ("Bob told Sarah that the bird caught the fish. It was slippery.", "it", "fish",   "bird",   2),
    ("Alex told John that the dog pushed the cat. It was frightened.", "it", "cat",    "dog",    2),
    ("Lisa told Anna that the ball hit the cup. It cracked.",          "it", "cup",    "ball",   2),
    ("Mary told Tom that the cat dropped the plate. It shattered.",    "it", "plate",  "cat",    2),
    ("Sarah told Mike that the car broke the window. It cracked.",     "it", "window", "car",    2),
    ("John told Bob that the fish bit the bird. It escaped.",          "it", "bird",   "fish",   2),
    ("Anna told Lisa that the dog chased the bird. It flew away.",     "it", "bird",   "dog",    2),
    ("Tom told Mary that the cup hit the plate. It fell.",             "it", "plate",  "cup",    2),
    ("Beth told Alex that the ball broke the tree. It collapsed.",     "it", "tree",   "ball",   2),
    ("Mike told Sarah that the cat pulled the dog. It ran away.",      "it", "dog",    "cat",    2),
    ("Bob told John that the car hit the cup. It shattered.",          "it", "cup",    "car",    2),
    ("Lisa told Beth that the bird dropped the fish. It sank.",        "it", "fish",   "bird",   2),
    ("Anna told Tom that the dog bit the cat. It cried.",              "it", "cat",    "dog",    2),
    ("Mary told Mike that the ball pushed the window. It cracked.",    "it", "window", "ball",   2),
    ("Sarah told Alex that the cat chased the bird. It was swift.",    "it", "cat",    "bird",   2),
]
assert len(GROUP1) == 20, f"Group1: {len(GROUP1)}"

# ── Group 2: depth 3 — moon-ring, relative clause inside embedded clause ──────
# "it" follows a doubly-embedded structure.
GROUP2: List[Tuple[str, str, str, str, int]] = [
    # ─ Given (4) ─
    ("John told Mary that the cat which was hungry chased the dog. It was fast.",        "it", "cat",    "dog",    3),
    ("Sarah said that the ball which was heavy broke the window. It shattered loudly.",  "it", "window", "ball",   3),
    ("Mike told Lisa that the dog which was angry bit the man. It barked loudly.",       "it", "dog",    "man",    3),
    ("Anna said that the car which was old hit the tree. It fell slowly.",               "it", "tree",   "car",    3),
    # ─ Generated (16) ─
    ("Tom told Beth that the cat which was clever caught the bird. It escaped quickly.", "it", "bird",   "cat",    3),
    ("Bob told Sarah that the dog which was strong pushed the man. It stumbled.",        "it", "man",    "dog",    3),
    ("Alex told John that the ball which was small hit the cup. It cracked noisily.",    "it", "cup",    "ball",   3),
    ("Lisa told Anna that the bird which was fast chased the fish. It dove deep.",       "it", "fish",   "bird",   3),
    ("Mary told Tom that the car which was large broke the gate. It collapsed.",         "it", "gate",   "car",    3),
    ("Sarah told Mike that the cat which was playful chased the dog. It hid away.",     "it", "dog",    "cat",    3),
    ("John told Bob that the fish which was large bit the bird. It flew off.",           "it", "bird",   "fish",   3),
    ("Anna told Lisa that the ball which was heavy hit the window. It broke.",           "it", "window", "ball",   3),
    ("Tom told Mary that the dog which was scared bit the cat. It yelped.",              "it", "cat",    "dog",    3),
    ("Beth told Alex that the car which was fast hit the tree. It toppled.",             "it", "tree",   "car",    3),
    ("Mike told Sarah that the bird which was trapped chased the cat. It bolted.",       "it", "cat",    "bird",   3),
    ("Bob told John that the dog which was lost found the cat. It hissed.",              "it", "cat",    "dog",    3),
    ("Lisa told Beth that the ball which was wet broke the cup. It shattered.",          "it", "cup",    "ball",   3),
    ("Anna told Tom that the cat which was thin pushed the dog. It fell over.",          "it", "dog",    "cat",    3),
    ("Mary told Mike that the fish which was slippery bit the bird. It screamed.",       "it", "bird",   "fish",   3),
    ("Sarah told Alex that the car which was blue hit the gate. It crumbled.",           "it", "gate",   "car",    3),
]
assert len(GROUP2) == 20, f"Group2: {len(GROUP2)}"

# ── Group 3: depth 4 — triply-embedded, deep subject identification ────────────
# No surface pronoun; question is which noun is the deep subject of the action.
# "correct" = the noun that WAS the deep patient (acted upon); "wrong" = the agent.
GROUP3: List[Tuple[str, str, str, str, int]] = [
    # ─ Given (3) ─
    ("John told Mary that the cat which the dog that Sarah owned had chased was injured.",   "injured_subj", "cat",  "dog",  4),
    ("Sarah said that the ball which the boy that Tom knew had thrown broke the window.",   "broke_subj",   "window","boy",  4),
    ("Mike told Lisa that the car which the man that Anna saw had driven hit the tree.",    "hit_subj",     "tree", "man",  4),
    # ─ Generated (17) ─
    ("Tom told Beth that the bird which the cat that Bob owned had chased was caught.",     "caught_subj",  "bird", "cat",  4),
    ("Anna told John that the fish which the dog that Lisa saw had chased was found.",      "found_subj",   "fish", "dog",  4),
    ("Bob told Sarah that the cup which the ball that Mary threw had hit was cracked.",     "cracked_subj", "cup",  "ball", 4),
    ("Alex told Tom that the gate which the car that Mike drove had hit was broken.",       "broken_subj",  "gate", "car",  4),
    ("Lisa told Anna that the dog which the cat that John saw had chased was tired.",       "tired_subj",   "dog",  "cat",  4),
    ("Mary told Bob that the bird which the fish that Sarah caught had seen was scared.",   "scared_subj",  "bird", "fish", 4),
    ("Sarah told Mike that the plate which the ball that Tom threw had hit was shattered.", "shattered_subj","plate","ball",4),
    ("John told Beth that the cat which the dog that Anna owned had chased was hurt.",      "hurt_subj",    "cat",  "dog",  4),
    ("Tom told Lisa that the window which the car that Bob drove had cracked was fixed.",   "fixed_subj",   "window","car", 4),
    ("Anna told Alex that the fish which the bird that Mary saw had eaten was gone.",       "gone_subj",    "fish", "bird", 4),
    ("Bob told Mary that the tree which the car that Sarah drove had hit was fallen.",      "fallen_subj",  "tree", "car",  4),
    ("Mike told John that the cup which the cat that Lisa owned had broken was replaced.",  "replaced_subj","cup",  "cat",  4),
    ("Beth told Tom that the dog which the bird that Alex saw had chased was panting.",     "panting_subj", "dog",  "bird", 4),
    ("Lisa told Sarah that the bird which the cat that Bob chased had met was free.",       "free_subj",    "bird", "cat",  4),
    ("Mary told Anna that the gate which the ball that Mike threw had hit was dented.",     "dented_subj",  "gate", "ball", 4),
    ("Sarah told Beth that the plate which the dog that John saw had broken was swept.",    "swept_subj",   "plate","dog",  4),
    ("Tom told Bob that the fish which the bird that Lisa saw had caught was eaten.",       "eaten_subj",   "fish", "bird", 4),
]
assert len(GROUP3) == 20, f"Group3: {len(GROUP3)}"

# ── Group 4: cross-depth — pronoun at moon level → sun subject ────────────────
# him/her refers to the outermost name (main-clause subject).
GROUP4: List[Tuple[str, str, str, str, str]] = [
    # ─ Given (2) ─
    ("John said that the cat which bit the dog scratched him.",          "him", "john",  "dog",   "cross"),
    ("Mary told Bob that the bird which the cat chased flew to her.",    "her", "mary",  "cat",   "cross"),
    # ─ Generated (18) ─
    ("Tom said that the dog which bit the bird surprised him.",          "him", "tom",   "bird",  "cross"),
    ("Lisa told Mike that the fish which the cat caught swam past her.", "her", "lisa",  "cat",   "cross"),
    ("Bob said that the cat which chased the bird startled him.",        "him", "bob",   "bird",  "cross"),
    ("Sarah told Alex that the dog which the ball hit ran to her.",      "her", "sarah", "ball",  "cross"),
    ("Mike said that the bird which caught the fish surprised him.",     "him", "mike",  "fish",  "cross"),
    ("Anna told John that the car which hit the tree scared her.",       "her", "anna",  "tree",  "cross"),
    ("Alex said that the dog which bit the man angered him.",            "him", "alex",  "man",   "cross"),
    ("Beth told Tom that the bird which the cat saw landed near her.",   "her", "beth",  "cat",   "cross"),
    ("John said that the cat which broke the cup upset him.",            "him", "john",  "cup",   "cross"),
    ("Mary told Bob that the ball which hit the window worried her.",    "her", "mary",  "window","cross"),
    ("Tom said that the fish which bit the bird surprised him.",         "him", "tom",   "bird",  "cross"),
    ("Lisa told Mike that the dog which the car almost hit ran to her.", "her", "lisa",  "car",   "cross"),
    ("Bob said that the bird which the cat chased landed near him.",     "him", "bob",   "cat",   "cross"),
    ("Sarah told Alex that the cat which broke the plate surprised her.","her", "sarah", "plate", "cross"),
    ("Mike said that the dog which bit the fish alarmed him.",           "him", "mike",  "fish",  "cross"),
    ("Anna told John that the bird which the cat chased flew to her.",   "her", "anna",  "cat",   "cross"),
    ("Alex said that the car which hit the gate startled him.",          "him", "alex",  "gate",  "cross"),
    ("Beth told Tom that the dog which chased the bird barked at her.",  "her", "beth",  "bird",  "cross"),
]
assert len(GROUP4) == 20, f"Group4: {len(GROUP4)}"

# ── Group 5: multi-pronoun — two pronouns in one sentence ─────────────────────
# Each item: (sentence, [(p1,correct1,wrong1), (p2,correct2,wrong2)], "multi")
# Correct if BOTH pronoun pairs resolve correctly.
GROUP5: List[Tuple[str, List[Tuple[str, str, str]], str]] = [
    # ─ Given (1) ─
    ("John told Mary that the cat chased the dog because it was hungry and he laughed.",
     [("it", "cat", "dog"), ("he", "john", "cat")], "multi"),
    # ─ Generated (19) ─
    ("Mary told Tom that the dog bit the bird but it escaped and she cried.",
     [("it", "bird", "dog"), ("she", "mary", "bird")], "multi"),
    ("Sarah told Mike that the ball broke the window and it shattered while she gasped.",
     [("it", "window", "ball"), ("she", "sarah", "window")], "multi"),
    ("John told Lisa that the cat caught the fish and it swam away while he watched.",
     [("it", "fish", "cat"), ("he", "john", "fish")], "multi"),
    ("Tom told Anna that the car hit the tree and it fell while he stopped.",
     [("it", "tree", "car"), ("he", "tom", "tree")], "multi"),
    ("Mike told Beth that the dog chased the bird because it was scared and he ran.",
     [("it", "bird", "dog"), ("he", "mike", "bird")], "multi"),
    ("Anna told Bob that the cat broke the cup because it was fragile and she laughed.",
     [("it", "cup", "cat"), ("she", "anna", "cup")], "multi"),
    ("Lisa told Chris that the ball hit the plate and it fell while she smiled.",
     [("it", "plate", "ball"), ("she", "lisa", "plate")], "multi"),
    ("Bob told Sarah that the dog bit the cat but it escaped and he apologized.",
     [("it", "cat", "dog"), ("he", "bob", "cat")], "multi"),
    ("Alex told Beth that the car hit the window and it shattered while he stopped.",
     [("it", "window", "car"), ("he", "alex", "window")], "multi"),
    ("Mary told Tom that the bird dropped the fish because it was heavy and she waited.",
     [("it", "fish", "bird"), ("she", "mary", "fish")], "multi"),
    ("John told Lisa that the dog bit the man because it was angry and he shouted.",
     [("it", "dog", "man"), ("he", "john", "dog")], "multi"),
    ("Sarah told Mike that the ball hit the tree and it swayed while she watched.",
     [("it", "tree", "ball"), ("she", "sarah", "tree")], "multi"),
    ("Tom told Anna that the cat chased the bird because it was fast and he cheered.",
     [("it", "cat", "bird"), ("he", "tom", "cat")], "multi"),
    ("Mike told Bob that the car broke the gate and it fell while he reversed.",
     [("it", "gate", "car"), ("he", "mike", "gate")], "multi"),
    ("Anna told John that the fish bit the bird but it escaped and she gasped.",
     [("it", "bird", "fish"), ("she", "anna", "bird")], "multi"),
    ("Lisa told Beth that the dog pushed the cat because it fell and she helped.",
     [("it", "cat", "dog"), ("she", "lisa", "cat")], "multi"),
    ("Bob told Mary that the ball broke the cup because it was thin and he gasped.",
     [("it", "cup", "ball"), ("he", "bob", "cup")], "multi"),
    ("Alex told Sarah that the cat chased the dog but it hid and he smiled.",
     [("it", "dog", "cat"), ("he", "alex", "dog")], "multi"),
    ("Beth told Tom that the bird chased the fish and it dove while she clapped.",
     [("it", "fish", "bird"), ("she", "beth", "fish")], "multi"),
]
assert len(GROUP5) == 20, f"Group5: {len(GROUP5)}"


# ══════════════════════════════════════════════════════════════════════════════
# Vocabulary + model loading
# ══════════════════════════════════════════════════════════════════════════════

def build_nested_vocab() -> Dict[str, int]:
    """Return the exact direct_train vocab (must match checkpoint embedding size)."""
    return rebuild_direct_vocab(max_vocab=5000)


def load_models(word2id, glove):
    vs     = len(word2id)
    solar  = SolarClassifier(vs,  glove).to(DEVICE, DTYPE)
    bilstm = BiLSTMClassifier(vs, glove).to(DEVICE, DTYPE)
    lstm   = LSTMClassifier(vs,   glove).to(DEVICE, DTYPE)
    for label, path, model in [
        ("solar",  "checkpoints/solar_direct_best.pt",  solar),
        ("bilstm", "checkpoints/bilstm_direct_best.pt", bilstm),
        ("lstm",   "checkpoints/lstm_direct_best.pt",   lstm),
    ]:
        p = Path(path)
        if p.exists():
            state = torch.load(p, map_location=DEVICE, weights_only=True)
            try:
                model.load_state_dict(state)
                print(f"  Loaded  {path}")
            except RuntimeError as e:
                print(f"  WARN    {path}: {e}  (using init weights)")
        else:
            print(f"  MISSING {path}  (using init weights)")
    return solar, bilstm, lstm


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def score_pair(model: nn.Module, ctx_ids: torch.Tensor, ans_id: int) -> float:
    """Score a (context + answer_token) sequence."""
    ids = torch.cat([ctx_ids, torch.tensor([ans_id], dtype=torch.long, device=DEVICE)])
    with torch.no_grad(), torch.autocast(DEVICE.type, dtype=DTYPE):
        return model(ids).item()


def eval_group_simple(
    models: Tuple,
    items:  List[Tuple],
    word2id: Dict[str, int],
) -> Dict[str, float]:
    """Evaluate groups 1-4: score(ctx + correct) > score(ctx + wrong)."""
    solar_m, bilstm_m, lstm_m = models
    results = {"solar": [], "bilstm": [], "lstm": []}
    for item in items:
        sentence, _, correct, wrong, _ = item
        ctx_ids  = encode(sentence, word2id).to(DEVICE)
        if ctx_ids.numel() == 0:
            continue
        corr_id = word2id.get(_normalize(correct), 0)
        wrong_id = word2id.get(_normalize(wrong),  0)
        for key, m in [("solar", solar_m), ("bilstm", bilstm_m), ("lstm", lstm_m)]:
            sc = score_pair(m, ctx_ids, corr_id)
            sw = score_pair(m, ctx_ids, wrong_id)
            results[key].append(sc > sw)
    return {k: 100.0 * sum(v) / max(len(v), 1) for k, v in results.items()}


def eval_group5(
    models:  Tuple,
    items:   List[Tuple],
    word2id: Dict[str, int],
) -> Dict[str, float]:
    """
    Evaluate group 5: both pronoun pairs must resolve correctly.
    A sentence counts as correct only if every (pronoun, correct, wrong) pair scores right.
    """
    solar_m, bilstm_m, lstm_m = models
    results = {"solar": [], "bilstm": [], "lstm": []}
    for sentence, pairs, _ in items:
        ctx_ids = encode(sentence, word2id).to(DEVICE)
        if ctx_ids.numel() == 0:
            continue
        for key, m in [("solar", solar_m), ("bilstm", bilstm_m), ("lstm", lstm_m)]:
            all_correct = True
            for _, correct, wrong in pairs:
                corr_id = word2id.get(_normalize(correct), 0)
                wrong_id = word2id.get(_normalize(wrong),  0)
                sc = score_pair(m, ctx_ids, corr_id)
                sw = score_pair(m, ctx_ids, wrong_id)
                if sc <= sw:
                    all_correct = False
                    break
            results[key].append(all_correct)
    return {k: 100.0 * sum(v) / max(len(v), 1) for k, v in results.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("NestedPronoun-100 Benchmark")
    print("5 groups × 20 items  |  cuda bfloat16  |  direct_train checkpoints")
    print("=" * 65)

    # Sanity-check dataset
    total = len(GROUP1) + len(GROUP2) + len(GROUP3) + len(GROUP4) + len(GROUP5)
    print(f"\n[0] Dataset: {total} total items")
    print(f"    Group 1 depth-2     : {len(GROUP1)} items")
    print(f"    Group 2 depth-3     : {len(GROUP2)} items")
    print(f"    Group 3 depth-4     : {len(GROUP3)} items")
    print(f"    Group 4 cross-depth : {len(GROUP4)} items")
    print(f"    Group 5 multi-pron  : {len(GROUP5)} items")

    # Vocabulary
    print("\n[1] Building vocabulary...")
    word2id = build_nested_vocab()
    print(f"    Vocab size: {len(word2id)}")

    # GloVe
    glove = None
    if Path(GLOVE_PATH).exists():
        print(f"\n[2] Loading GloVe 300d...")
        glove = load_glove(GLOVE_PATH, word2id, d=300)
        print(f"    Matrix: {glove.shape}")
    else:
        print(f"\n[2] GloVe not found — using random embeddings")

    # Load models
    print("\n[3] Loading models from checkpoints...")
    solar_m, bilstm_m, lstm_m = load_models(word2id, glove)
    solar_m.eval(); bilstm_m.eval(); lstm_m.eval()
    models = (solar_m, bilstm_m, lstm_m)

    # Evaluate
    print("\n[4] Evaluating...\n")
    g1 = eval_group_simple(models, GROUP1, word2id)
    g2 = eval_group_simple(models, GROUP2, word2id)
    g3 = eval_group_simple(models, GROUP3, word2id)
    g4 = eval_group_simple(models, GROUP4, word2id)
    g5 = eval_group5(models, GROUP5, word2id)

    groups = [
        ("Group 1  Depth 2 (planet)", g1),
        ("Group 2  Depth 3 (moon)  ", g2),
        ("Group 3  Depth 4 (sub)   ", g3),
        ("Group 4  Cross-depth     ", g4),
        ("Group 5  Multi-pronoun   ", g5),
    ]

    # ── Per-group results ─────────────────────────────────────────────────
    print("=" * 65)
    print(f"  {'Group':<28} {'Solar Ring':>12} {'BiLSTM':>8} {'LSTM':>8}")
    print("-" * 65)
    for label, res in groups:
        print(f"  {label:<28} {res['solar']:>11.1f}% {res['bilstm']:>7.1f}% {res['lstm']:>7.1f}%")

    # ── Overall ───────────────────────────────────────────────────────────
    all_results = [g1, g2, g3, g4, g5]
    overall = {
        k: sum(r[k] for r in all_results) / len(all_results)
        for k in ("solar", "bilstm", "lstm")
    }
    print("-" * 65)
    print(
        f"  {'Overall':<28} {overall['solar']:>11.1f}% "
        f"{overall['bilstm']:>7.1f}% {overall['lstm']:>7.1f}%"
    )
    print("=" * 65)

    # ── Depth degradation analysis ────────────────────────────────────────
    print("\nDepth degradation (Solar Ring vs LSTM):")
    for label, res in groups[:3]:
        delta = res["solar"] - res["lstm"]
        sign  = "+" if delta >= 0 else ""
        print(f"  {label.strip():<30}  Solar={res['solar']:.1f}%  "
              f"LSTM={res['lstm']:.1f}%  Δ={sign}{delta:.1f}%")

    print()


if __name__ == "__main__":
    main()
