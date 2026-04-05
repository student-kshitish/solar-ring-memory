"""Benchmark 1: Low-resource training with tiny subsets.

Train all three classifiers on 10/25/50/100/200 items each and measure
accuracy on all 90 Winograd schemas.
"""

import sys, os, random, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

from solar_ring.model        import SolarRingModel
from solar_ring.config       import D_MODEL
from solar_ring.glove_loader import load_glove
from baseline.vanilla_lstm   import VanillaLSTM
from baseline.bilstm         import BiLSTM
from benchmarks.direct_train import (
    SolarClassifier, BiLSTMClassifier, LSTMClassifier,
    build_generated_pairs, build_vocab, encode,
    eval_schemas, evaluate_classifier,
)
from benchmarks.winograd_full import WINOGRAD_SCHEMAS

GLOVE_PATH = "data/glove.6B.300d.txt"
TRAIN_SIZES = [10, 25, 50, 100, 200]
EPOCHS = 20


def train_small(model, train_items, eval_schemas_list, word2id, epochs=20):
    encoded = []
    for text, label in train_items:
        ids = encode(text, word2id)
        if ids.numel() > 0:
            encoded.append((ids, float(label)))

    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCELoss()
    best_acc = 0.0
    best_state = None

    for ep in range(epochs):
        model.train()
        for ids, label in encoded:
            ids = ids.to(DEVICE)
            lbl = torch.tensor(label, device=DEVICE)
            opt.zero_grad()
            with torch.autocast(DEVICE.type, dtype=DTYPE):
                score = model(ids)
            loss = criterion(score.float(), lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        acc = eval_schemas(model, eval_schemas_list, word2id)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def run():
    print("\n" + "=" * 62)
    print("BENCHMARK 1: Low-Resource Training")
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
    word2id    = build_vocab(all_texts + wino_texts, max_vocab=5000)

    glove = None
    if Path(GLOVE_PATH).exists():
        glove = load_glove(GLOVE_PATH, word2id, d=300)
        print(f"  GloVe loaded: {glove.shape}")
    else:
        print("  WARNING: GloVe not found — random init")

    eval_schemas_list = schemas[70:]   # 20 held-out schemas
    vs = len(word2id)

    results = {}  # train_size -> {model: acc}

    for train_size in TRAIN_SIZES:
        print(f"\n  --- Train size = {train_size} ---")
        subset = all_items[:train_size]

        # Fresh models each time
        solar_clf  = SolarClassifier(vs,  glove).to(DEVICE, DTYPE)
        bilstm_clf = BiLSTMClassifier(vs, glove).to(DEVICE, DTYPE)
        lstm_clf   = LSTMClassifier(vs,   glove).to(DEVICE, DTYPE)

        solar_clf  = train_small(solar_clf,  subset, eval_schemas_list, word2id, EPOCHS)
        bilstm_clf = train_small(bilstm_clf, subset, eval_schemas_list, word2id, EPOCHS)
        lstm_clf   = train_small(lstm_clf,   subset, eval_schemas_list, word2id, EPOCHS)

        solar_acc,  _ = evaluate_classifier(solar_clf,  "SolarRing",  word2id)
        bilstm_acc, _ = evaluate_classifier(bilstm_clf, "BiLSTM",     word2id)
        lstm_acc,   _ = evaluate_classifier(lstm_clf,   "LSTM",       word2id)

        results[train_size] = {
            "solar": solar_acc,
            "bilstm": bilstm_acc,
            "lstm": lstm_acc,
        }
        print(f"    N={train_size:4d}: Solar={solar_acc:.1%}  BiLSTM={bilstm_acc:.1%}  LSTM={lstm_acc:.1%}")

    print("\n" + "=" * 62)
    print(f"{'Train N':>8} | {'Solar Ring':>10} | {'BiLSTM':>8} | {'LSTM':>8}")
    print("-" * 46)
    for n, r in results.items():
        print(f"{n:>8} | {r['solar']:>9.1%} | {r['bilstm']:>7.1%} | {r['lstm']:>7.1%}")
    print("=" * 62)

    return results


if __name__ == "__main__":
    run()
