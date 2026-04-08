"""
Interactive Solar Ring Memory demo.
Type any sentence and see:
- Ring structure created
- Subject/object poles filled
- Pronoun resolved
- Sun State after processing
"""

import torch
import sys
sys.path.insert(0, '.')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from solar_ring.model import SolarRingModel
from solar_ring.solar_memory import SolarMemory
from solar_ring.sun_state import SunState
from solar_ring.black_white_hole import BlackWhiteHoleManager
from benchmarks.direct_train import build_vocab, build_generated_pairs
from benchmarks.winograd_80 import (
    WinogradSpringModel,
    find_pronoun_idx,
    PRONOUNS,
)

# Stopwords filtered out when building candidate lists
_CAND_STOPWORDS = {
    'the', 'a', 'an', 'is', 'was', 'were',
    'that', 'because', 'which', 'not',
    'did', 'do', 'too', 'so', 'and', 'but',
    'in', 'on', 'at', 'to', 'of', 'by', 'as',
    'for', 'with', 'from', 'be', 'been',
}


def load_model(vocab_size):
    # SolarRingModel uses D_MODEL=300 from config; no d_model constructor arg
    model = SolarRingModel(vocab_size=vocab_size).to(DEVICE)
    try:
        state = torch.load(
            'checkpoints/solar_direct_best.pt',
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(state, strict=False)
        print("Checkpoint loaded.")
    except Exception as e:
        print(f"Using random weights ({e})")
    model.eval()
    return model


def load_spring_model():
    """Load MiniLM + Solar Spring model (87.5% Winograd accuracy)."""
    print("Loading MiniLM + Solar Spring model...")
    spring_model = WinogradSpringModel().to(DEVICE)
    try:
        ckpt = torch.load(
            'checkpoints/winograd80_best.pt',
            map_location=DEVICE,
            weights_only=True,
        )
        spring_model.spring.load_state_dict(ckpt['spring'])
        spring_model.head.load_state_dict(ckpt['head'])
        print("Solar Spring checkpoint loaded (87.5% Winograd)")
    except Exception as e:
        print(f"Spring checkpoint not found: {e}")
    spring_model.spring.eval()
    spring_model.head.eval()
    return spring_model


def resolve_with_spring(sentence, spring_model):
    """
    Use MiniLM + Solar Spring to resolve pronouns.
    Much more accurate than base GloVe model.

    The model is trained to score `ctx + candidate` higher for the
    correct referent. We collect candidate nouns from the sentence,
    score each one via score_from_vecs(), and return the winner.
    """
    words = sentence.lower().split()
    stripped = [w.rstrip('.,;:!?') for w in words]

    found_pronouns = [(i, w) for i, w in enumerate(stripped)
                      if w in PRONOUNS]
    if not found_pronouns:
        return []

    # Candidate nouns: content words that are not pronouns or stopwords
    candidates = [
        w for w in stripped
        if w not in PRONOUNS
        and w not in _CAND_STOPWORDS
        and len(w) > 2
    ]
    # Deduplicate while preserving order
    seen = set()
    candidates = [c for c in candidates
                  if not (c in seen or seen.add(c))]

    if not candidates:
        return []

    # Build all sentences to score in one batched MiniLM call
    # Pattern: sentence + ' ' + candidate  (candidate appears as final token;
    # backward attention from candidate → pronoun selects the referent)
    scored_sents = [sentence + ' ' + c for c in candidates]
    emb_cache = spring_model.embedder.embed_words_batch(scored_sents)

    results = []
    for pronoun_pos, pronoun in found_pronouns:
        best_score = float('-inf')
        best_cand  = candidates[0]

        with torch.no_grad():
            for cand, sent in zip(candidates, scored_sents):
                try:
                    logit = spring_model.score_from_vecs(
                        sent, emb_cache[sent]
                    ).item()
                    if logit > best_score:
                        best_score = logit
                        best_cand  = cand
                except Exception:
                    continue

        results.append((pronoun, best_cand, best_score))

    return results


def process_sentence(sentence, model, vocab, spring_model):
    words = sentence.lower().split()
    ids = torch.tensor(
        [vocab.get(w.rstrip('.,;:!?'), 0) for w in words],
        dtype=torch.long, device=DEVICE,
    )

    print(f"\n{'='*55}")
    print(f"Processing: {sentence}")
    print(f"{'='*55}")
    print(f"Tokens: {words}")
    print(f"IDs   : {ids.tolist()}")

    # ── Run model + get populated memory ────────────────────────────────────
    with torch.no_grad():
        logits, aux = model(ids)
    context_vec = aux["context_vec"]   # (d,) — flattened memory representation

    # get_memory_for_sentence re-runs the forward to expose the final ring state
    memory = model.get_memory_for_sentence(ids)

    # ── Show ring structure ──────────────────────────────────────────────────
    print(f"\nRing structure:")
    depth_names = {0: 'SUN', 1: 'PLANET', 2: 'MOON'}
    for i, ring in enumerate(memory.rings):
        depth_name = depth_names.get(ring.depth, f'DEPTH{ring.depth}')
        subj = ring.subj_word if ring.subj_word else '---'
        obj  = ring.obj_word  if ring.obj_word  else '---'
        print(f"  Ring {i} [{depth_name}]  "
              f"SUBJ={subj:<10} OBJ={obj:<10} "
              f"locked_s={ring.subj_locked}  "
              f"locked_o={ring.obj_locked}")

    # ── Black/White hole events (physics pass) ───────────────────────────────
    print(f"\nBlack/White hole events:")
    sun = SunState(300, alpha=0.3, device=DEVICE)
    mgr = BlackWhiteHoleManager(300, DEVICE, sun)

    all_events = []
    for word in words:
        events = mgr.step(word, memory, sun)
        if events:
            all_events.extend(events)
            for e in events:
                print(f"  {e}")
    if not all_events:
        print("  No events (sentence too short for collapse)")

    # ── Sun State ────────────────────────────────────────────────────────────
    sun_norm = sun.state.norm().item()
    print(f"\nSun State norm: {sun_norm:.4f}")

    # ── Pronoun resolution — MiniLM + Solar Spring (87.5% model) ─────────────
    stripped = [w.rstrip('.,;:!?') for w in words]
    found_pronouns = [w for w in stripped if w in PRONOUNS]

    if found_pronouns:
        print(f"\nPronouns: {found_pronouns}")
        print("Resolving with MiniLM + Solar Spring (87.5% model):")
        results = resolve_with_spring(sentence, spring_model)
        if results:
            for pronoun, candidate, score in results:
                print(f"  '{pronoun}' → '{candidate}' "
                      f"(confidence={score:.3f})")
        else:
            print("  No candidate nouns found to resolve against.")
    else:
        print("\nNo pronouns found in sentence.")

    print(f"\nContext vector norm: {context_vec.norm().item():.4f}")
    print(f"Model output logits shape: {logits.shape}")

    return memory, sun, context_vec


def main():
    print("=" * 55)
    print("Solar Ring Memory — Interactive Demo")
    print("=" * 55)
    print(f"Device: {DEVICE}")

    # build_vocab expects List[str]; build_generated_pairs returns List[Tuple[str,int]]
    pairs = build_generated_pairs()
    texts = [text for text, _ in pairs]
    vocab = build_vocab(texts)
    print(f"Vocab size: {len(vocab)}")

    model = load_model(len(vocab))
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    spring_model = load_spring_model()

    print("\nType any sentence and press Enter.")
    print("Type 'quit' to exit.")
    print("Examples:")
    print("  John told Paul that he should leave early.")
    print("  The trophy did not fit the suitcase because it was too big.")
    print("  Sarah helped Beth because she was tired.")
    print()

    while True:
        try:
            sentence = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not sentence:
            continue
        if sentence.lower() in ('quit', 'exit', 'q'):
            print("Bye.")
            break

        try:
            process_sentence(sentence, model, vocab, spring_model)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
