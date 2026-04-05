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
from solar_ring.solar_memory import SolarMemory
from solar_ring.config import ROLE_SUBJ, ROLE_OBJ, ROLE_VERB, ROLE_OTHER
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


def _build_sun_state_for_sent1(sent1: str, glove, word2id) -> "SolarMemory":
    """
    Process sent1 tokens into a fresh SolarMemory, then call end_clause()
    so the Sun State absorbs sentence-1 entity information.
    Returns the memory with a warm Sun State.
    """
    from solar_ring.config import D_MODEL
    memory = SolarMemory(device=DEVICE, dtype=DTYPE)

    # Simple heuristic POS: first noun→SUBJ, last noun→OBJ, verbs→VERB
    words = sent1.split()
    noun_indices = []
    for i, w in enumerate(words):
        wl = w.lower().rstrip(".,!?")
        # crude noun detector: not a function word
        if wl not in {"the","a","an","and","or","but","that","which","who","was","is","are","were","had","has","have"}:
            noun_indices.append(i)

    for i, w in enumerate(words):
        wl = w.lower().rstrip(".,!?")
        vid = word2id.get(wl, 0)
        if glove is not None and vid < glove.shape[0]:
            vec = torch.tensor(glove[vid], dtype=DTYPE, device=DEVICE)
        else:
            vec = torch.zeros(300, dtype=DTYPE, device=DEVICE)

        # Assign role by position heuristic
        if noun_indices and i == noun_indices[0]:
            role = ROLE_SUBJ
        elif noun_indices and i == noun_indices[-1]:
            role = ROLE_OBJ
        elif wl in {"told","asked","gave","helped","hired","praised","wrote","met","bought","baked","fixed","trained","launched","repaired","promoted","landed","interviewed"}:
            role = ROLE_VERB
        else:
            role = ROLE_OTHER
        memory.process_token(vec, role, token_text=wl)

    memory.end_clause()
    return memory


def evaluate_cross_sun_state(model, word2id, glove, sun_weight: float = 0.3):
    """
    Sun State enhanced evaluation.
    For each pair:
      1. Process sent1 → end_clause() → warm Sun State
      2. Score sent2+referent normally (base classifier)
      3. Compute resonance of correct vs wrong referent with Sun State
      4. Final score = base_score + sun_weight * resonance_score
    """
    correct = 0
    for sent1, sent2, pronoun, corr, wrong in CROSS_SENT_PAIRS:
        memory = _build_sun_state_for_sent1(sent1, glove, word2id)

        # Base classifier scores on full combined text
        combined = sent1 + " " + sent2
        sc_base = score_pair(model, combined, corr,  word2id)
        sw_base = score_pair(model, combined, wrong, word2id)

        # Sun State resonance scores for candidate referents
        def ref_resonance(ref_word):
            wl = ref_word.lower()
            vid = word2id.get(wl, 0)
            if glove is not None and vid < glove.shape[0]:
                vec = torch.tensor(glove[vid], dtype=torch.float32, device=DEVICE)
            else:
                return 0.0
            return memory.get_sun_resonance(vec)

        res_corr  = ref_resonance(corr)
        res_wrong = ref_resonance(wrong)

        sc_final = sc_base + sun_weight * res_corr
        sw_final = sw_base + sun_weight * res_wrong

        if sc_final > sw_final:
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

    # Sun State enhanced evaluation (Solar Ring only — SR owns this feature)
    print("\n  -- Sun State enhanced (Solar Ring) --")
    acc_sun = evaluate_cross_sun_state(solar_clf, word2id, glove, sun_weight=0.3)
    results["Solar Ring + Sun"] = acc_sun
    print(f"  Solar Ring + Sun State: {acc_sun:.1%}  ({int(acc_sun * len(CROSS_SENT_PAIRS))}/{len(CROSS_SENT_PAIRS)} correct)")

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
    print(f"{'Model':<22} | {'Accuracy':>10} | {'Correct/Total':>14}")
    print("-" * 54)
    for name, acc in results.items():
        n = int(acc * len(CROSS_SENT_PAIRS))
        marker = " ← Sun State" if "Sun" in name else ""
        print(f"{name:<22} | {acc:>9.1%} | {n:>6}/{len(CROSS_SENT_PAIRS)}{marker}")
    baseline = results.get("Solar Ring", 0)
    sun_acc  = results.get("Solar Ring + Sun", 0)
    delta    = sun_acc - baseline
    print("-" * 54)
    print(f"  Sun State improvement: {delta:+.1%}  ({baseline:.1%} → {sun_acc:.1%})")
    print("=" * 62)

    return results


if __name__ == "__main__":
    run()
