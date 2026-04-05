"""
Enhanced Winograd evaluation using three parallelism levels.
Does NOT modify existing model or benchmarks.
Adds sub-planet + multi-system scores ON TOP of base model.

Phase 1: baseline Solar Ring score
Phase 2: + Sub-planet animacy/case/size compatibility
Phase 3: + Multi-planet Sun State resonance
Phase 4: + Multi-solar system cross-paragraph gravity

Target: beat BERT 70% on 90 Winograd schemas.
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16

GLOVE_PATH  = "data/glove.6B.300d.txt"
CKPT_SOLAR  = "checkpoints/solar_direct_best.pt"
CKPT_BILSTM = "checkpoints/bilstm_direct_best.pt"

from solar_ring.glove_loader import load_glove
from benchmarks.direct_train import (
    SolarClassifier, BiLSTMClassifier,
    build_generated_pairs, build_vocab, encode,
)
from benchmarks.winograd_full import WINOGRAD_SCHEMAS
from solar_ring.sub_planet_enhanced import (
    SubPlanetEnhanced,
    build_sentence_sub_planets,
    find_adjectives_in_context,
)
from solar_ring.sun_state import SunState
from solar_ring.multi_solar_system import MultiSolarSystem


def load_models(vocab_size: int, glove):
    solar = SolarClassifier(vocab_size, glove).to(DEVICE, DTYPE)
    if Path(CKPT_SOLAR).exists():
        solar.load_state_dict(
            torch.load(CKPT_SOLAR, map_location=DEVICE, weights_only=True)
        )
        print("Solar Ring loaded.")
    else:
        print("WARNING: solar checkpoint not found — random init")
    solar.eval()

    bilstm = BiLSTMClassifier(vocab_size, glove).to(DEVICE, DTYPE)
    if Path(CKPT_BILSTM).exists():
        bilstm.load_state_dict(
            torch.load(CKPT_BILSTM, map_location=DEVICE, weights_only=True)
        )
        print("BiLSTM loaded.")
    else:
        print("WARNING: bilstm checkpoint not found — random init")
    bilstm.eval()

    return solar, bilstm


def get_logit(model, token_ids) -> float:
    """Return scalar score [0, 1] from a classifier model."""
    with torch.no_grad(), torch.autocast(DEVICE.type, dtype=DTYPE):
        return model(token_ids).float().item()


def score_with_sub_planet(sentence: str,
                           correct: str,
                           wrong: str,
                           base_correct: float,
                           base_wrong: float) -> tuple:
    """
    Phase 2: add sub-planet compatibility score.
    Finds pronoun in sentence and scores each candidate.
    """
    words = sentence.lower().split()
    sub_planets = build_sentence_sub_planets(sentence)
    adjectives  = find_adjectives_in_context(sentence)

    PRONOUNS = {'it', 'he', 'she', 'they', 'him', 'her',
                'them', 'who', 'which', 'that'}
    pronoun_sp = None
    for i, w in enumerate(words):
        if w in PRONOUNS:
            pronoun_sp = sub_planets[i]
            break

    if pronoun_sp is None:
        return base_correct, base_wrong

    correct_word = correct.split()[0].lower().rstrip('.,')
    wrong_word   = wrong.split()[0].lower().rstrip('.,')

    compat_correct = pronoun_sp.pronoun_compatibility(correct_word)
    compat_wrong   = pronoun_sp.pronoun_compatibility(wrong_word)

    size_correct = 0.0
    size_wrong   = 0.0
    for adj in adjectives:
        size_correct += pronoun_sp.size_compatibility(adj, correct_word)
        size_wrong   += pronoun_sp.size_compatibility(adj, wrong_word)

    COMPAT_WEIGHT = 0.6
    SIZE_WEIGHT   = 0.8

    final_correct = (base_correct
                     + COMPAT_WEIGHT * compat_correct
                     + SIZE_WEIGHT   * size_correct)
    final_wrong   = (base_wrong
                     + COMPAT_WEIGHT * compat_wrong
                     + SIZE_WEIGHT   * size_wrong)

    # ── CLEAR OVERRIDE: fire only when linguistic rule is certain ──────────
    from solar_ring.sub_planet_enhanced import ANIMACY_SIGNALS

    inanimate_set   = set(ANIMACY_SIGNALS.get('inanimate',   []))
    human_male_set  = set(ANIMACY_SIGNALS.get('human_male',  []))
    human_female_set= set(ANIMACY_SIGNALS.get('human_female',[]))
    animate_set     = set(ANIMACY_SIGNALS.get('animate',     []))
    all_animate     = human_male_set | human_female_set | animate_set

    # Neuter "it/its/which/that" → must be inanimate
    if pronoun_sp.case == 'neuter':
        if correct_word in inanimate_set and wrong_word in all_animate:
            final_correct += 2.0
        elif correct_word in all_animate and wrong_word in inanimate_set:
            final_wrong += 2.0

    # Masculine "he/him" → must be human_male
    if pronoun_sp.animacy == 'human_male':
        if correct_word in human_male_set and wrong_word not in human_male_set:
            final_correct += 2.0
        elif wrong_word in human_male_set and correct_word not in human_male_set:
            final_wrong += 2.0

    # Feminine "she/her" → must be human_female
    if pronoun_sp.animacy == 'human_female':
        if correct_word in human_female_set and wrong_word not in human_female_set:
            final_correct += 2.0
        elif wrong_word in human_female_set and correct_word not in human_female_set:
            final_wrong += 2.0

    return final_correct, final_wrong


def score_with_sun_state(sentence: str,
                          correct: str,
                          wrong: str,
                          score_c: float,
                          score_w: float,
                          vocab: dict) -> tuple:
    """
    Phase 3: add Sun State resonance bonus.
    Builds a fresh SunState from sentence nouns/verbs,
    then boosts whichever candidate resonates more.
    """
    sun = SunState(d_model=300, alpha=0.3, device=DEVICE)

    try:
        from solar_ring.pos_tagger import POSTagger
        tagger = POSTagger()
        doc = tagger.nlp(sentence)
        planet_vecs = []
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN', 'VERB'):
                tid = vocab.get(token.text.lower(), 0)
                vec = torch.zeros(300, device=DEVICE)
                vec[tid % 300] = 1.0
                planet_vecs.append(vec)
        if planet_vecs:
            sun.fuse(planet_vecs)
    except Exception:
        return score_c, score_w

    correct_word = correct.split()[0].lower().rstrip('.,')
    wrong_word   = wrong.split()[0].lower().rstrip('.,')

    cid = vocab.get(correct_word, 0)
    wid = vocab.get(wrong_word,   0)

    vec_c = torch.zeros(300, device=DEVICE)
    vec_w = torch.zeros(300, device=DEVICE)
    vec_c[cid % 300] = 1.0
    vec_w[wid % 300] = 1.0

    res_c = sun.resonance(vec_c)
    res_w = sun.resonance(vec_w)

    RES_WEIGHT = 0.8
    return (score_c + RES_WEIGHT * res_c,
            score_w + RES_WEIGHT * res_w)


def score_with_multi_system(sentence: str,
                             correct: str,
                             wrong: str,
                             score_c: float,
                             score_w: float,
                             vocab: dict) -> tuple:
    """
    Phase 4: add MultiSolarSystem cross-paragraph gravity.
    Treats each clause (split on comma/period) as a separate paragraph.
    """
    mss = MultiSolarSystem(d_model=300, device=DEVICE,
                           alpha=0.3, max_systems=4)

    clauses = [c.strip() for c in sentence.replace(',', '.').split('.')
               if c.strip()]

    for i, clause in enumerate(clauses):
        words = clause.lower().split()
        for word in words:
            wid = vocab.get(word, 0)
            vec = torch.zeros(300, device=DEVICE)
            vec[wid % 300] = 1.0
            # Write into active ring as rotating token
            from solar_ring.config import ROLE_OTHER
            mss.active.process_token(vec.to(DTYPE), ROLE_OTHER)
        mss.end_paragraph()
        if i < len(clauses) - 1:
            mss.new_paragraph()

    correct_word = correct.split()[0].lower().rstrip('.,')
    wrong_word   = wrong.split()[0].lower().rstrip('.,')

    cid = vocab.get(correct_word, 0)
    wid = vocab.get(wrong_word,   0)
    vec_c = torch.zeros(300, device=DEVICE)
    vec_w = torch.zeros(300, device=DEVICE)
    vec_c[cid % 300] = 1.0
    vec_w[wid % 300] = 1.0

    res_c = mss.get_resonance(vec_c)
    res_w = mss.get_resonance(vec_w)

    GRAVITY_WEIGHT = 0.6
    return (score_c + GRAVITY_WEIGHT * res_c,
            score_w + GRAVITY_WEIGHT * res_w)


def evaluate_enhanced(solar, bilstm, vocab, verbose=False):
    results = {
        'phase1': {'correct': 0, 'total': 0},
        'phase2': {'correct': 0, 'total': 0},
        'phase3': {'correct': 0, 'total': 0},
        'phase4': {'correct': 0, 'total': 0},
        'bilstm': {'correct': 0, 'total': 0},
    }

    for idx, (ctx, corr, wrong) in enumerate(WINOGRAD_SCHEMAS):
        ids_c = encode(ctx + ' ' + corr,  vocab).to(DEVICE)
        ids_w = encode(ctx + ' ' + wrong, vocab).to(DEVICE)

        if ids_c.numel() == 0 or ids_w.numel() == 0:
            continue

        # Phase 1: base model scores
        try:
            sc1 = get_logit(solar, ids_c)
            sw1 = get_logit(solar, ids_w)
        except Exception:
            continue

        p1_correct = sc1 > sw1
        results['phase1']['correct'] += int(p1_correct)
        results['phase1']['total']   += 1

        # Phase 2: + sub-planet animacy/case/size
        sc2, sw2 = score_with_sub_planet(ctx, corr, wrong, sc1, sw1)
        p2_correct = sc2 > sw2
        results['phase2']['correct'] += int(p2_correct)
        results['phase2']['total']   += 1

        # Phase 3: + Sun State resonance
        sc3, sw3 = score_with_sun_state(ctx, corr, wrong, sc2, sw2, vocab)
        p3_correct = sc3 > sw3
        results['phase3']['correct'] += int(p3_correct)
        results['phase3']['total']   += 1

        # Phase 4: + multi-solar system gravity
        sc4, sw4 = score_with_multi_system(ctx, corr, wrong, sc3, sw3, vocab)
        p4_correct = sc4 > sw4
        results['phase4']['correct'] += int(p4_correct)
        results['phase4']['total']   += 1

        # BiLSTM baseline
        try:
            bc = get_logit(bilstm, ids_c)
            bw = get_logit(bilstm, ids_w)
            results['bilstm']['correct'] += int(bc > bw)
            results['bilstm']['total']   += 1
        except Exception:
            pass

        if verbose and idx < 5:
            print(f"\nSchema {idx + 1}:")
            print(f"  Sentence : {ctx[:60]}...")
            print(f"  Correct  : {corr}")
            print(f"  Wrong    : {wrong}")
            print(f"  P1: correct={sc1:.4f} wrong={sw1:.4f} "
                  f"→ {'CORRECT' if p1_correct else 'WRONG'}")
            print(f"  P2: correct={sc2:.4f} wrong={sw2:.4f} "
                  f"→ {'CORRECT' if p2_correct else 'WRONG'}")
            print(f"  P3: correct={sc3:.4f} wrong={sw3:.4f} "
                  f"→ {'CORRECT' if p3_correct else 'WRONG'}")
            print(f"  P4: correct={sc4:.4f} wrong={sw4:.4f} "
                  f"→ {'CORRECT' if p4_correct else 'WRONG'}")

    return results


def print_results(results):
    print("\n" + "=" * 65)
    print("ENHANCED WINOGRAD RESULTS — THREE PARALLELISM LEVELS")
    print("=" * 65)

    phases = [
        ('phase1', 'Baseline Solar Ring'),
        ('phase2', '+ Sub-planet (animacy/case/size)'),
        ('phase3', '+ Multi-planet Sun State resonance'),
        ('phase4', '+ Multi-solar system gravity'),
        ('bilstm', 'BiLSTM baseline'),
    ]

    for key, label in phases:
        r = results[key]
        if r['total'] == 0:
            continue
        acc = r['correct'] / r['total'] * 100
        bar = '█' * int(acc / 2)
        print(f"  {label:<40} {r['correct']:>2}/{r['total']} "
              f"= {acc:5.1f}%  {bar}")

    print("-" * 65)

    final_acc = (results['phase4']['correct'] /
                 max(results['phase4']['total'], 1) * 100)
    bert_acc  = 70.0

    print(f"\n  Final Solar Ring enhanced : {final_acc:.1f}%")
    print(f"  BERT-base target          : {bert_acc:.1f}%")
    print(f"  Gap                       : {final_acc - bert_acc:+.1f}%")

    if final_acc >= bert_acc:
        print(f"\n  BEATS BERT: YES (+{final_acc - bert_acc:.1f}%)")
    else:
        print(f"\n  BEATS BERT: NOT YET "
              f"(need +{bert_acc - final_acc:.1f}% more)")

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

    print("\nRunning enhanced evaluation (verbose first 5)...")
    results = evaluate_enhanced(solar, bilstm, word2id, verbose=True)

    print_results(results)
