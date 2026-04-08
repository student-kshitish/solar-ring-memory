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


def process_sentence(sentence, model, vocab):
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
    sun  = SunState(300, alpha=0.3, device=DEVICE)
    mgr  = BlackWhiteHoleManager(300, DEVICE, sun)

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

    # ── Pronoun resolution ───────────────────────────────────────────────────
    PRONOUNS = {'it', 'he', 'she', 'they', 'him', 'her',
                'them', 'who', 'which', 'that'}
    found_pronouns = [w.rstrip('.,;:!?') for w in words
                      if w.rstrip('.,;:!?') in PRONOUNS]

    if found_pronouns:
        print(f"\nPronouns found: {found_pronouns}")
        for pronoun in found_pronouns:
            pron_id  = vocab.get(pronoun, 0)
            pron_vec = model.embedding.weight[pron_id]

            best_score = -1.0
            best_word  = 'unknown'

            for w in words:
                wc = w.rstrip('.,;:!?')
                if wc in PRONOUNS:
                    continue
                wid = vocab.get(wc, 0)
                if wid == 0:
                    continue
                w_vec = model.embedding.weight[wid]
                score = torch.cosine_similarity(
                    pron_vec.unsqueeze(0),
                    w_vec.unsqueeze(0),
                ).item()
                if score > best_score:
                    best_score = score
                    best_word  = wc

            print(f"  '{pronoun}' → '{best_word}' (score={best_score:.3f})")
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

    print("\nType any sentence and press Enter.")
    print("Type 'quit' to exit.")
    print("Examples:")
    print("  John told Mary that the cat chased the dog.")
    print("  Sarah helped Beth because she was tired.")
    print("  The trophy did not fit because it was too big.")
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
            process_sentence(sentence, model, vocab)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
