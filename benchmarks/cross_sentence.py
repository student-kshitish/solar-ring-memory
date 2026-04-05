"""Benchmark 4: Cross-sentence coreference resolution.

20 two-sentence pairs. Test whether models can resolve anaphora
that spans sentence boundaries by concatenating both sentences.
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

# (sent1, sent2, pronoun, correct_referent, wrong_referent)
CROSS_SENT_PAIRS = [
    ("John bought a trophy.", "It was made of gold.", "it", "trophy", "John"),
    ("Mary hired an assistant.", "She started on Monday.", "she", "Mary", "assistant"),
    ("The dog chased the cat.", "It ran into the garden.", "it", "cat", "dog"),
    ("Paul won the race.", "He received a medal.", "he", "Paul", "race"),
    ("Anna baked a cake.", "It smelled wonderful.", "it", "cake", "Anna"),
    ("The teacher praised the student.", "She had worked very hard.", "she", "student", "teacher"),
    ("Tom fixed the car.", "It runs perfectly now.", "it", "car", "Tom"),
    ("Sarah gave John a gift.", "He was very happy.", "he", "John", "Sarah"),
    ("The scientist published a paper.", "It was widely cited.", "it", "paper", "scientist"),
    ("Mike trained for months.", "He finally won the championship.", "he", "Mike", "championship"),
    ("The company launched a product.", "It became very popular.", "it", "product", "company"),
    ("Lisa met Emma at the conference.", "She gave a great talk.", "she", "Emma", "Lisa"),
    ("The coach trained the athlete.", "She broke the record.", "she", "athlete", "coach"),
    ("Bob wrote a book.", "It was published last year.", "it", "book", "Bob"),
    ("Rachel interviewed the candidate.", "She was very impressive.", "she", "candidate", "Rachel"),
    ("David repaired the machine.", "It works again now.", "it", "machine", "David"),
    ("The manager promoted the worker.", "He deserved it.", "he", "worker", "manager"),
    ("Kate bought flowers.", "They were for her mother.", "they", "flowers", "Kate"),
    ("The pilot landed the plane.", "It touched down smoothly.", "it", "plane", "pilot"),
    ("Emma trained the puppy.", "It learned quickly.", "it", "puppy", "Emma"),
]


def score_pair(model, text, referent, word2id):
    ids = encode(text + " " + referent, word2id).to(DEVICE)
    if ids.numel() == 0:
        return 0.5
    with torch.no_grad(), torch.autocast(DEVICE.type, dtype=DTYPE):
        return model(ids).item()


def evaluate_cross(model, word2id):
    correct = 0
    for sent1, sent2, pronoun, corr, wrong in CROSS_SENT_PAIRS:
        combined = sent1 + " " + sent2
        sc = score_pair(model, combined, corr,  word2id)
        sw = score_pair(model, combined, wrong, word2id)
        if sc > sw:
            correct += 1
    return correct / len(CROSS_SENT_PAIRS)


def run():
    print("\n" + "=" * 62)
    print("BENCHMARK 4: Cross-Sentence Coreference")
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
        acc = evaluate_cross(model, word2id)
        results[name] = acc
        print(f"  {name}: {acc:.1%}  ({int(acc * len(CROSS_SENT_PAIRS))}/{len(CROSS_SENT_PAIRS)} correct)")

    # Detailed trace for first 5 pairs
    print("\n  Detailed trace (first 5 pairs):")
    print(f"  {'Pair':<40} | {'Solar':>6} | {'BiLSTM':>6} | {'LSTM':>6}")
    print("  " + "-" * 70)
    for sent1, sent2, pronoun, corr, wrong in CROSS_SENT_PAIRS[:5]:
        combined = sent1 + " " + sent2
        sr_ok  = "✓" if score_pair(solar_clf,  combined, corr, word2id) > score_pair(solar_clf,  combined, wrong, word2id) else "✗"
        bl_ok  = "✓" if score_pair(bilstm_clf, combined, corr, word2id) > score_pair(bilstm_clf, combined, wrong, word2id) else "✗"
        ls_ok  = "✓" if score_pair(lstm_clf,   combined, corr, word2id) > score_pair(lstm_clf,   combined, wrong, word2id) else "✗"
        pair_str = (sent1 + " / " + sent2)[:39]
        print(f"  {pair_str:<40} | {sr_ok:>6} | {bl_ok:>6} | {ls_ok:>6}")

    print("\n" + "=" * 62)
    print(f"{'Model':<12} | {'Accuracy':>10} | {'Correct/Total':>14}")
    print("-" * 44)
    for name, acc in results.items():
        n = int(acc * len(CROSS_SENT_PAIRS))
        print(f"{name:<12} | {acc:>9.1%} | {n:>6}/{len(CROSS_SENT_PAIRS)}")
    print("=" * 62)

    return results


if __name__ == "__main__":
    run()
