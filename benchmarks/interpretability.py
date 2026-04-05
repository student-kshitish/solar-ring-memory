"""Benchmark 2: Interpretability trace for Solar Ring vs BiLSTM.

For 10 Winograd schemas, print a detailed trace showing:
- Token sequence
- Classifier score for correct vs wrong referent
- Whether resolution was correct
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
    SolarClassifier, BiLSTMClassifier,
    build_generated_pairs, build_vocab, encode,
)
from benchmarks.winograd_full import WINOGRAD_SCHEMAS, _pronoun_category

GLOVE_PATH = "data/glove.6B.300d.txt"
CKPT_SOLAR  = "checkpoints/solar_direct_best.pt"
CKPT_BILSTM = "checkpoints/bilstm_direct_best.pt"
N_TRACE = 10


def load_models(word2id, glove):
    vs = len(word2id)
    solar_clf  = SolarClassifier(vs,  glove).to(DEVICE, DTYPE)
    bilstm_clf = BiLSTMClassifier(vs, glove).to(DEVICE, DTYPE)

    if Path(CKPT_SOLAR).exists():
        solar_clf.load_state_dict(torch.load(CKPT_SOLAR,  weights_only=True, map_location=DEVICE))
        print(f"  Loaded {CKPT_SOLAR}")
    else:
        print(f"  WARNING: {CKPT_SOLAR} not found — using random init")

    if Path(CKPT_BILSTM).exists():
        bilstm_clf.load_state_dict(torch.load(CKPT_BILSTM, weights_only=True, map_location=DEVICE))
        print(f"  Loaded {CKPT_BILSTM}")
    else:
        print(f"  WARNING: {CKPT_BILSTM} not found — using random init")

    solar_clf.eval()
    bilstm_clf.eval()
    return solar_clf, bilstm_clf


def score_pair(model, ctx, referent, word2id):
    ids = encode(ctx + " " + referent, word2id).to(DEVICE)
    if ids.numel() == 0:
        return 0.5
    with torch.no_grad(), torch.autocast(DEVICE.type, dtype=DTYPE):
        return model(ids).item()


def run():
    print("\n" + "=" * 62)
    print("BENCHMARK 2: Interpretability Trace")
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

    solar_clf, bilstm_clf = load_models(word2id, glove)

    # Pick 10 diverse schemas for the trace
    trace_schemas = WINOGRAD_SCHEMAS[:N_TRACE]

    solar_correct  = 0
    bilstm_correct = 0

    print(f"\n{'=' * 70}")
    for i, (ctx, corr, wrong) in enumerate(trace_schemas):
        cat = _pronoun_category(ctx)
        tokens = ctx.split()
        token_preview = " ".join(tokens[:12]) + ("..." if len(tokens) > 12 else "")

        sc_corr  = score_pair(solar_clf,  ctx, corr,  word2id)
        sc_wrong = score_pair(solar_clf,  ctx, wrong, word2id)
        bl_corr  = score_pair(bilstm_clf, ctx, corr,  word2id)
        bl_wrong = score_pair(bilstm_clf, ctx, wrong, word2id)

        solar_ok  = sc_corr > sc_wrong
        bilstm_ok = bl_corr > bl_wrong
        solar_correct  += int(solar_ok)
        bilstm_correct += int(bilstm_ok)

        sr_label = "CORRECT" if solar_ok  else "WRONG"
        bl_label = "CORRECT" if bilstm_ok else "WRONG"

        print(f"\nSchema {i+1:2d}/{N_TRACE}  [{cat}]")
        print(f"  Sentence : {ctx}")
        print(f"  Tokens   : [{token_preview}]  ({len(tokens)} tokens)")
        print(f"  Correct  : '{corr}'")
        print(f"  Wrong    : '{wrong}'")
        print(f"  Solar Ring  → correct score={sc_corr:.4f}  wrong score={sc_wrong:.4f}  → {sr_label}")
        print(f"  BiLSTM      → correct score={bl_corr:.4f}  wrong score={bl_wrong:.4f}  → {bl_label}")
        print("-" * 70)

    solar_acc  = solar_correct  / N_TRACE
    bilstm_acc = bilstm_correct / N_TRACE

    print(f"\n  Accuracy on {N_TRACE} schemas:")
    print(f"    Solar Ring : {solar_correct}/{N_TRACE} = {solar_acc:.1%}")
    print(f"    BiLSTM     : {bilstm_correct}/{N_TRACE} = {bilstm_acc:.1%}")
    print("=" * 62)

    return {"solar": solar_acc, "bilstm": bilstm_acc}


if __name__ == "__main__":
    run()
