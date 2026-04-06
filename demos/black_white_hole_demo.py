"""
Demo showing black hole and white hole mechanics
on real sentences.
"""

import torch
import sys
sys.path.insert(0, '.')

from solar_ring.solar_memory import SolarMemory
from solar_ring.sun_state import SunState
from solar_ring.black_white_hole import BlackWhiteHoleManager
from solar_ring.config import (
    ROLE_SUBJ, ROLE_OBJ, ROLE_VERB,
    ROLE_CONJ, ROLE_DET, ROLE_OTHER
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = 300

def demo():
    print("="*60)
    print("Black Hole / White Hole Demo")
    print("="*60)

    # Scenario: contradicted belief triggers black hole
    # New pronoun triggers white hole
    sentences = [
        ("John",    "SUBJ"),
        ("told",    "VERB"),
        ("Mary",    "OBJ"),
        ("that",    "CONJ"),
        ("the",     "DET"),
        ("cat",     "SUBJ"),
        ("not",     "OTHER"),   # ← negation → black hole trigger
        ("chased",  "VERB"),
        ("the",     "DET"),
        ("dog",     "OBJ"),
        ("but",     "CONJ"),
        ("it",      "OTHER"),   # ← orphan pronoun → white hole
        ("escaped", "VERB"),
    ]

    sun    = SunState(D, alpha=0.3, device=DEVICE)
    memory = SolarMemory(device=DEVICE)
    manager = BlackWhiteHoleManager(D, DEVICE, sun)

    ROLE_MAP = {
        'SUBJ':  ROLE_SUBJ,
        'OBJ':   ROLE_OBJ,
        'VERB':  ROLE_VERB,
        'CONJ':  ROLE_CONJ,
        'DET':   ROLE_DET,
        'OTHER': ROLE_OTHER,
    }

    print("\nProcessing tokens:")
    print(f"{'Token':12} {'Role':8} {'Events'}")
    print("-"*60)

    for word, role in sentences:
        vec     = torch.randn(D, device=DEVICE)
        role_id = ROLE_MAP.get(role, ROLE_OTHER)
        memory.process_token(vec, role_id)

        events    = manager.step(word, memory, sun)
        event_str = ' | '.join(events) if events else ''
        print(f"{word:12} {role:8} {event_str}")

    print("\n" + "="*60)
    print("Final state:")
    print(f"  {manager.summary()}")
    print(f"  Collapsed rings: {manager.collapsed_indices}")
    print(f"  Sun State norm : {sun.state.norm():.4f}")
    print(f"  Spawned rings  : {manager.white_hole.spawned_rings}")
    print("="*60)

    print("\nWhat happened:")
    print("  'not' → BLACK HOLE triggered on active ring")
    print("         Ring collapsed → Hawking radiation → Sun State")
    print("  'it'  → WHITE HOLE triggered (orphan pronoun)")
    print("         Placeholder ring spawned from Sun State seed")
    print("         Ring gravitates toward most likely antecedent")

if __name__ == "__main__":
    demo()
