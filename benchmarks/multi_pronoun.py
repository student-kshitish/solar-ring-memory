"""Benchmark 3: Multi-pronoun resolution.

20 sentences with two pronouns each. Evaluate how well each model
resolves BOTH referents correctly.
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

from solar_ring.glove_loader import load_glove
from benchmarks.direct_train import (
    SolarClassifier, BiLSTMClassifier, LSTMClassifier,
    build_generated_pairs, build_vocab, encode,
)
from benchmarks.winograd_full import WINOGRAD_SCHEMAS

GLOVE_PATH  = "data/glove.6B.300d.txt"
CKPT_SOLAR  = "checkpoints/solar_direct_best.pt"
CKPT_BILSTM = "checkpoints/bilstm_direct_best.pt"
CKPT_LSTM   = "checkpoints/lstm_direct_best.pt"

# (sentence, pronoun1, correct_ref1, wrong_ref1, pronoun2, correct_ref2, wrong_ref2)
MULTI_PRONOUN_SENTENCES = [
    ("John told Mary that he would help her with the work.",
     "he", "John", "Mary", "her", "Mary", "John"),
    ("Sarah asked Tom if he could drive her to the station.",
     "he", "Tom", "Sarah", "her", "Sarah", "Tom"),
    ("The cat chased the dog until it caught it near the fence.",
     "it", "cat", "dog", "it", "dog", "cat"),
    ("Alice gave Bob the book and he thanked her for it.",
     "he", "Bob", "Alice", "her", "Alice", "Bob"),
    ("Michael told David he was promoted and David congratulated him.",
     "he", "Michael", "David", "him", "David", "Michael"),
    ("Emma helped Julia because she was tired and Julia needed her support.",
     "she", "Emma", "Julia", "her", "Emma", "Julia"),
    ("The doctor examined the patient and he told him to rest.",
     "he", "doctor", "patient", "him", "patient", "doctor"),
    ("Rachel asked Lisa if she could borrow her notes.",
     "she", "Rachel", "Lisa", "her", "Lisa", "Rachel"),
    ("Tom hired Alex because he was skilled and paid him well.",
     "he", "Alex", "Tom", "him", "Alex", "Tom"),
    ("The teacher praised the student because she had worked hard and she deserved it.",
     "she", "student", "teacher", "she", "student", "teacher"),
    ("Anna told Beth that she had passed and Beth hugged her happily.",
     "she", "Anna", "Beth", "her", "Anna", "Beth"),
    ("Paul called James because he needed advice and James gave him some.",
     "he", "Paul", "James", "him", "Paul", "James"),
    ("The manager fired the worker because he was late and he apologized.",
     "he", "worker", "manager", "he", "worker", "manager"),
    ("Sara met Kate at the party and she introduced her to everyone.",
     "she", "Sara", "Kate", "her", "Kate", "Sara"),
    ("The coach praised the athlete after she trained hard and she won the race.",
     "she", "athlete", "coach", "she", "athlete", "coach"),
    ("Mark helped Nick because he was struggling and Nick thanked him.",
     "he", "Nick", "Mark", "him", "Mark", "Nick"),
    ("The nurse told the doctor he was needed and he went immediately.",
     "he", "doctor", "nurse", "he", "doctor", "nurse"),
    ("Lucy invited Karen because she liked her cooking.",
     "she", "Lucy", "Karen", "her", "Karen", "Lucy"),
    ("The professor failed the student because he had cheated and he regretted it.",
     "he", "student", "professor", "he", "student", "professor"),
    ("Chris met Jamie and he lent him his umbrella.",
     "he", "Chris", "Jamie", "him", "Jamie", "Chris"),
]


def score_pair(model, ctx, referent, word2id):
    ids = encode(ctx + " " + referent, word2id).to(DEVICE)
    if ids.numel() == 0:
        return 0.5
    with torch.no_grad(), torch.autocast(DEVICE.type, dtype=DTYPE):
        return model(ids).item()


def evaluate_multi(model, word2id):
    both = one = neither = 0
    for entry in MULTI_PRONOUN_SENTENCES:
        ctx, p1, c1, w1, p2, c2, w2 = entry
        s_c1 = score_pair(model, ctx, c1, word2id)
        s_w1 = score_pair(model, ctx, w1, word2id)
        s_c2 = score_pair(model, ctx, c2, word2id)
        s_w2 = score_pair(model, ctx, w2, word2id)
        ok1 = s_c1 > s_w1
        ok2 = s_c2 > s_w2
        if ok1 and ok2:
            both += 1
        elif ok1 or ok2:
            one += 1
        else:
            neither += 1
    n = len(MULTI_PRONOUN_SENTENCES)
    return both / n, one / n, neither / n


def run():
    print("\n" + "=" * 62)
    print("BENCHMARK 3: Multi-Pronoun Resolution")
    print("=" * 62)

    random.seed(42)
    schemas = list(WINOGRAD_SCHEMAS)
    random.shuffle(schemas)

    wino_items = []
    for ctx, corr, wrong in schemas[:70]:
        wino_items.append((ctx + " " + corr, 1))
        wino_items.append((ctx + " " + wrong, 0))

    gen_items = build_generated_pairs()
    all_items = wino_items + gen_items
    all_texts  = [t for t, _ in all_items]
    wino_texts = [t for ctx, c, w in WINOGRAD_SCHEMAS for t in (ctx, c, w)]
    word2id = build_vocab(all_texts + wino_texts, max_vocab=5000)

    glove = None
    if Path(GLOVE_PATH).exists():
        glove = load_glove(GLOVE_PATH, word2id, d=300)

    vs = len(word2id)
    solar_clf  = SolarClassifier(vs,  glove).to(DEVICE, DTYPE)
    bilstm_clf = BiLSTMClassifier(vs, glove).to(DEVICE, DTYPE)
    lstm_clf   = LSTMClassifier(vs,   glove).to(DEVICE, DTYPE)

    for clf, ckpt, name in [
        (solar_clf,  CKPT_SOLAR,  "Solar"),
        (bilstm_clf, CKPT_BILSTM, "BiLSTM"),
        (lstm_clf,   CKPT_LSTM,   "LSTM"),
    ]:
        if Path(ckpt).exists():
            clf.load_state_dict(torch.load(ckpt, weights_only=True, map_location=DEVICE))
            print(f"  Loaded {ckpt}")
        else:
            print(f"  WARNING: {ckpt} not found — random init")
        clf.eval()

    results = {}
    for model, name in [(solar_clf, "Solar Ring"), (bilstm_clf, "BiLSTM"), (lstm_clf, "LSTM")]:
        both, one, neither = evaluate_multi(model, word2id)
        results[name] = (both, one, neither)
        print(f"  {name}: both={both:.1%}  one={one:.1%}  neither={neither:.1%}")

    print("\n" + "=" * 62)
    print(f"{'Model':<12} | {'Both%':>6} | {'One%':>6} | {'Neither%':>9}")
    print("-" * 44)
    for name, (b, o, n) in results.items():
        print(f"{name:<12} | {b:>5.1%} | {o:>5.1%} | {n:>8.1%}")
    print("=" * 62)

    return results


if __name__ == "__main__":
    run()
