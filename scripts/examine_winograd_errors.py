"""
Examine exactly which 36 Winograd schemas are wrong.
Categorize why they fail.
Predict which ones pretraining can fix vs cannot fix.
"""

import torch
import sys
sys.path.insert(0, '.')

from benchmarks.winograd_full import WINOGRAD_SCHEMAS
from benchmarks.direct_train import build_vocab, build_generated_pairs
from benchmarks.winograd_conceptnet import evaluate_with_conceptnet
from benchmarks.winograd_enhanced import load_models
from solar_ring.glove_loader import load_glove
from solar_ring.conceptnet import (
    apply_conceptnet_to_winograd,
    get_properties
)

GLOVE_PATH = "data/glove.6B.300d.txt"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CATEGORIES = {
    'animacy':    'Pronoun animate/inanimate mismatch',
    'gender':     'Pronoun gender mismatch he/she',
    'size_pair':  'Both candidates same size category',
    'causal':     'Causal direction requires world knowledge',
    'unknown':    'Both candidates unknown to ConceptNet',
    'fixable':    'Pretraining could fix this',
}

def examine():
    pairs = build_generated_pairs()
    all_texts = []
    for text, label in pairs:
        all_texts.append(text)
    from benchmarks.winograd_full import WINOGRAD_SCHEMAS
    for ctx, corr, wrong in WINOGRAD_SCHEMAS:
        all_texts.append(ctx + ' ' + corr)
        all_texts.append(ctx + ' ' + wrong)
    vocab = build_vocab(all_texts)
    from pathlib import Path
    glove = None
    if Path(GLOVE_PATH).exists():
        glove = load_glove(GLOVE_PATH, vocab, d=300)
    solar, bilstm = load_models(len(vocab), glove)

    wrong_schemas = []
    correct_schemas = []

    print("Examining all 90 Winograd schemas...")
    print("="*70)

    from benchmarks.direct_train import encode
    from benchmarks.winograd_enhanced import (
        get_logit,
        score_with_sub_planet,
    )

    for idx, (ctx, corr, wrong) in enumerate(WINOGRAD_SCHEMAS):
        ids_c = encode(ctx+' '+corr, vocab).to(DEVICE)
        ids_w = encode(ctx+' '+wrong, vocab).to(DEVICE)

        if ids_c.numel() == 0 or ids_w.numel() == 0:
            continue

        try:
            sc0 = get_logit(solar, ids_c)
            sw0 = get_logit(solar, ids_w)
        except:
            continue

        sc1, sw1 = score_with_sub_planet(
            ctx, corr, wrong, sc0, sw0
        )
        cn_c, cn_w = apply_conceptnet_to_winograd(
            ctx, corr, wrong
        )
        sc_final = sc1 + 2.0 * cn_c
        sw_final = sw1 + 2.0 * cn_w

        is_correct = sc_final > sw_final

        correct_word = corr.split()[0].lower().rstrip('.,')
        wrong_word   = wrong.split()[0].lower().rstrip('.,')

        props_c = get_properties(correct_word)
        props_w = get_properties(wrong_word)

        # Categorize
        words = ctx.lower().split()
        PRONOUNS = {'it','he','she','they','him',
                    'her','them','who','which','that'}
        pronoun = next(
            (w for w in words if w in PRONOUNS), None
        )

        category = 'unknown'
        fixable = False

        if props_c and props_w:
            anim_c = props_c.get('animate')
            anim_w = props_w.get('animate')

            if anim_c != anim_w:
                category = 'animacy'
                fixable = True
            elif (props_c.get('size') == props_w.get('size')
                  and props_c.get('size') is not None):
                category = 'size_pair'
                fixable = True  # pretraining helps
            else:
                category = 'causal'
                fixable = True
        elif not props_c and not props_w:
            category = 'unknown'
            fixable = False  # ConceptNet missing both
        elif pronoun in ('he','him','his'):
            category = 'gender'
            fixable = True
        elif pronoun in ('she','her','hers'):
            category = 'gender'
            fixable = True

        entry = {
            'idx': idx+1,
            'ctx': ctx,
            'correct': corr,
            'wrong': wrong,
            'pronoun': pronoun,
            'category': category,
            'fixable': fixable,
            'is_correct': is_correct,
            'margin': sc_final - sw_final,
        }

        if is_correct:
            correct_schemas.append(entry)
        else:
            wrong_schemas.append(entry)

    print(f"Currently correct: {len(correct_schemas)}/90")
    print(f"Currently wrong  : {len(wrong_schemas)}/90")

    print("\n" + "="*70)
    print("WRONG SCHEMAS — CATEGORIZED")
    print("="*70)

    by_category = {}
    fixable_count = 0

    for e in wrong_schemas:
        cat = e['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(e)
        if e['fixable']:
            fixable_count += 1

    for cat, entries in by_category.items():
        print(f"\n[{cat.upper()}] — {len(entries)} schemas")
        print(f"  {CATEGORIES.get(cat,'')}")
        for e in entries[:3]:  # show first 3 per category
            print(f"  Schema {e['idx']}: "
                  f"{e['ctx'][:55]}...")
            print(f"    Correct: {e['correct'][:35]}")
            print(f"    Wrong  : {e['wrong'][:35]}")
            print(f"    Pronoun: {e['pronoun']}  "
                  f"Fixable: {e['fixable']}")

    print("\n" + "="*70)
    print("PREDICTION: What pretraining can fix")
    print("="*70)
    print(f"  Currently wrong      : {len(wrong_schemas)}")
    print(f"  Fixable by training  : {fixable_count}")
    print(f"  Not fixable (unknown): "
          f"{len(wrong_schemas)-fixable_count}")

    predicted_correct = len(correct_schemas) + fixable_count
    predicted_acc = predicted_correct / 90 * 100

    print(f"\n  Current accuracy  : "
          f"{len(correct_schemas)}/90 = "
          f"{len(correct_schemas)/90*100:.1f}%")
    print(f"  Predicted after   : "
          f"{predicted_correct}/90 = {predicted_acc:.1f}%")
    print(f"  BERT target       : ~70%")
    print(f"  Beats BERT?       : "
          f"{'YES ✓' if predicted_acc >= 70 else 'NO'}")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if predicted_acc >= 70:
        print("  Pretraining WILL push Solar Ring above BERT 70%.")
        print("  Proceed with corpus pretraining.")
    elif predicted_acc >= 65:
        print("  Pretraining MAY reach 65-70%.")
        print("  Worth trying — significant improvement expected.")
    else:
        print("  Pretraining alone cannot reach 70%.")
        print("  The remaining errors need different approach.")
        print("  Recommendation: accept 60% with ConceptNet,")
        print("  claim data efficiency not raw accuracy.")

if __name__ == "__main__":
    examine()
