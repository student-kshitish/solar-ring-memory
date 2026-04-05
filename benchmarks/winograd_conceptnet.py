"""
Winograd evaluation with ConceptNet knowledge injection.
Combines base Solar Ring model + all parallelism phases + ConceptNet facts.
Target: 80%+ on 90 Winograd schemas.
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16

GLOVE_PATH  = "data/glove.6B.300d.txt"
CONCEPTNET_WEIGHT = 2.0

from solar_ring.glove_loader import load_glove
from benchmarks.direct_train import (
    build_generated_pairs, build_vocab, encode,
)
from benchmarks.winograd_full import WINOGRAD_SCHEMAS
from benchmarks.winograd_enhanced import (
    load_models, get_logit,
    score_with_sub_planet, score_with_sun_state,
    score_with_multi_system,
)
from solar_ring.conceptnet import apply_conceptnet_to_winograd


def evaluate_with_conceptnet(solar, bilstm, vocab, verbose=False):
    results = {
        'base':       {'correct': 0, 'total': 0},
        'sub_planet': {'correct': 0, 'total': 0},
        'sun_state':  {'correct': 0, 'total': 0},
        'multi_sys':  {'correct': 0, 'total': 0},
        'conceptnet': {'correct': 0, 'total': 0},
        'bilstm':     {'correct': 0, 'total': 0},
    }

    for idx, (ctx, corr, wrong) in enumerate(WINOGRAD_SCHEMAS):
        ids_c = encode(ctx + ' ' + corr,  vocab).to(DEVICE)
        ids_w = encode(ctx + ' ' + wrong, vocab).to(DEVICE)

        if ids_c.numel() == 0 or ids_w.numel() == 0:
            continue

        # Phase 0: base model
        try:
            sc0 = get_logit(solar, ids_c)
            sw0 = get_logit(solar, ids_w)
        except Exception:
            continue

        results['base']['correct'] += int(sc0 > sw0)
        results['base']['total']   += 1

        # Phase 1: + sub-planet animacy/case/size
        sc1, sw1 = score_with_sub_planet(ctx, corr, wrong, sc0, sw0)
        results['sub_planet']['correct'] += int(sc1 > sw1)
        results['sub_planet']['total']   += 1

        # Phase 2: + Sun State resonance
        sc2, sw2 = score_with_sun_state(ctx, corr, wrong, sc1, sw1, vocab)
        results['sun_state']['correct'] += int(sc2 > sw2)
        results['sun_state']['total']   += 1

        # Phase 3: + multi-solar system gravity
        sc3, sw3 = score_with_multi_system(ctx, corr, wrong, sc2, sw2, vocab)
        results['multi_sys']['correct'] += int(sc3 > sw3)
        results['multi_sys']['total']   += 1

        # Phase 4: + ConceptNet knowledge boost
        cn_c, cn_w = apply_conceptnet_to_winograd(ctx, corr, wrong)
        sc4 = sc3 + CONCEPTNET_WEIGHT * cn_c
        sw4 = sw3 + CONCEPTNET_WEIGHT * cn_w
        results['conceptnet']['correct'] += int(sc4 > sw4)
        results['conceptnet']['total']   += 1

        # BiLSTM baseline
        try:
            bc = get_logit(bilstm, ids_c)
            bw = get_logit(bilstm, ids_w)
            results['bilstm']['correct'] += int(bc > bw)
            results['bilstm']['total']   += 1
        except Exception:
            pass

        if verbose and idx < 10:
            cn_fired = cn_c != 0.0 or cn_w != 0.0
            print(f"\nSchema {idx + 1}: {ctx[:55]}...")
            print(f"  Correct    : {corr[:45]}")
            print(f"  Wrong      : {wrong[:45]}")
            print(f"  Base       : c={sc0:.3f} w={sw0:.3f} "
                  f"→ {'✓' if sc0 > sw0 else '✗'}")
            print(f"  ConceptNet : c_adj={cn_c:+.2f} w_adj={cn_w:+.2f} "
                  f"fired={'YES' if cn_fired else 'no'}")
            print(f"  Final      : c={sc4:.3f} w={sw4:.3f} "
                  f"→ {'✓' if sc4 > sw4 else '✗'}")

    return results


def print_conceptnet_results(results):
    print("\n" + "=" * 65)
    print("WINOGRAD — SOLAR RING + CONCEPTNET KNOWLEDGE")
    print("=" * 65)

    rows = [
        ('base',       'Baseline Solar Ring'),
        ('sub_planet', '+ Sub-planet parallelism'),
        ('sun_state',  '+ Sun State resonance'),
        ('multi_sys',  '+ Multi-solar system'),
        ('conceptnet', '+ ConceptNet knowledge'),
        ('bilstm',     'BiLSTM baseline'),
    ]

    for key, label in rows:
        r = results[key]
        if r['total'] == 0:
            continue
        acc = r['correct'] / r['total'] * 100
        bar = '█' * int(acc / 2)
        print(f"  {label:<38} {r['correct']:>2}/{r['total']} "
              f"= {acc:5.1f}%  {bar}")

    print("-" * 65)

    final = (results['conceptnet']['correct'] /
             max(results['conceptnet']['total'], 1) * 100)

    print(f"\n  Final Solar Ring + ConceptNet : {final:.1f}%")
    print(f"  BERT-base                     : ~70.0%")
    print(f"  Target                        : 80.0%")
    print(f"  Gap to BERT                   : {final - 70:+.1f}%")
    print(f"  Gap to target                 : {final - 80:+.1f}%")

    if final >= 80.0:
        print(f"\n  TARGET REACHED: {final:.1f}% >= 80%")
    elif final >= 70.0:
        print(f"\n  BEATS BERT: {final:.1f}% >= 70%")
    else:
        print(f"\n  Gap remaining: need +{80 - final:.1f}% more")

    print("=" * 65)


if __name__ == "__main__":
    print("Loading vocab...")
    random.seed(42)
    schemas = list(WINOGRAD_SCHEMAS)
    random.shuffle(schemas)

    wino_items = []
    for ctx, corr, wrong in schemas[:70]:
        wino_items.append((ctx + ' ' + corr,  1))
        wino_items.append((ctx + ' ' + wrong, 0))

    gen_items  = build_generated_pairs()
    all_items  = wino_items + gen_items
    all_texts  = [t for t, _ in all_items]
    wino_texts = [t for ctx, c, w in WINOGRAD_SCHEMAS for t in (ctx, c, w)]
    word2id    = build_vocab(all_texts + wino_texts, max_vocab=5000)
    print(f"Vocab size: {len(word2id)}")

    glove = None
    if Path(GLOVE_PATH).exists():
        glove = load_glove(GLOVE_PATH, word2id, d=300)

    print("Loading models...")
    solar, bilstm = load_models(len(word2id), glove)

    print("\nRunning evaluation (verbose first 10)...")
    results = evaluate_with_conceptnet(solar, bilstm, word2id, verbose=True)

    print_conceptnet_results(results)
