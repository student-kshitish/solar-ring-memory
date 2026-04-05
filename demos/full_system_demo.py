"""Full Solar Ring system demo — Sun State, Gravity Gate, Sub-planet."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from solar_ring.config import D_MODEL, ROLE_SUBJ, ROLE_OBJ, ROLE_VERB, ROLE_OTHER
from solar_ring.solar_memory import SolarMemory
from solar_ring.sun_state import SunState
from solar_ring.gravity_gate import GravityGate, POS_MASS
from solar_ring.sub_planet import SubPlanet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

torch.manual_seed(42)

# ── Fake GloVe-style embeddings (300d) ────────────────────────────────────────
# Use fixed random vectors so results are reproducible
VOCAB = {
    "john":    torch.randn(D_MODEL),
    "told":    torch.randn(D_MODEL),
    "mary":    torch.randn(D_MODEL),
    "that":    torch.randn(D_MODEL),
    "the":     torch.randn(D_MODEL),
    "cat":     torch.randn(D_MODEL),
    "chased":  torch.randn(D_MODEL),
    "dog":     torch.randn(D_MODEL),
    "because": torch.randn(D_MODEL),
    "it":      torch.randn(D_MODEL),
    "was":     torch.randn(D_MODEL),
    "hungry":  torch.randn(D_MODEL),
    "escaped": torch.randn(D_MODEL),
    "quickly": torch.randn(D_MODEL),
    "purple":  torch.randn(D_MODEL),  # control — never in either sentence
}
# Normalize all embeddings to unit sphere
for k in VOCAB:
    VOCAB[k] = VOCAB[k] / (VOCAB[k].norm() + 1e-8)

def embed(word: str) -> torch.Tensor:
    return VOCAB.get(word.lower(), torch.zeros(D_MODEL)).to(DEVICE)

# ── POS assignments for sentence 1 ────────────────────────────────────────────
# "John told Mary that the cat chased the dog because it was hungry."
SENT1 = [
    ("John",    "SUBJ",  ROLE_SUBJ),
    ("told",    "VERB",  ROLE_VERB),
    ("Mary",    "OBJ",   ROLE_OBJ),
    ("that",    "CONJ",  ROLE_OTHER),
    ("the",     "DET",   ROLE_OTHER),
    ("cat",     "SUBJ",  ROLE_SUBJ),
    ("chased",  "VERB",  ROLE_VERB),
    ("the",     "DET",   ROLE_OTHER),
    ("dog",     "OBJ",   ROLE_OBJ),
    ("because", "CONJ",  ROLE_OTHER),
    ("it",      "SUBJ",  ROLE_SUBJ),
    ("was",     "VERB",  ROLE_VERB),
    ("hungry",  "ADJ",   ROLE_OTHER),
]

# "The dog escaped quickly."
SENT2 = [
    ("The",     "DET",   ROLE_OTHER),
    ("dog",     "SUBJ",  ROLE_SUBJ),
    ("escaped", "VERB",  ROLE_VERB),
    ("quickly", "ADV",   ROLE_OTHER),
]

# ── Build system components ────────────────────────────────────────────────────
memory = SolarMemory(device=DEVICE, dtype=DTYPE)
gravity_gate = GravityGate(D_MODEL).to(DEVICE)
gravity_gate.eval()

# Sub-planet for tracking "dog"
dog_sub = SubPlanet(D_MODEL, device=DEVICE)

print("=" * 62)
print("FULL SOLAR RING SYSTEM DEMO")
print("=" * 62)
print(f'\nSentence 1: "John told Mary that the cat chased the dog because it was hungry."')
print(f'Sentence 2: "The dog escaped quickly."\n')

# ── Process sentence 1 ────────────────────────────────────────────────────────
print("Processing sentence 1...")
for word, pos_type, role_id in SENT1:
    vec = embed(word)
    memory.process_token(vec.to(DTYPE), role_id, token_text=word)
    if word.lower() == "dog":
        dog_sub.update_parallel(vec, word, pos_type)

# End of clause 1 → fuse into Sun
memory.end_clause()
sun_norm_s1 = memory.sun_state.state.norm().item()
print(f"  Sun state norm after sentence 1: {sun_norm_s1:.4f}")
print(f"  Sun fusions so far:              {memory.sun_state.age}")

# ── Process sentence 2 ────────────────────────────────────────────────────────
print("\nProcessing sentence 2...")
for word, pos_type, role_id in SENT2:
    vec = embed(word)
    memory.process_token(vec.to(DTYPE), role_id, token_text=word)
    if word.lower() == "dog":
        dog_sub.update_parallel(vec, word, pos_type)

memory.end_clause()
sun_norm_s2 = memory.sun_state.state.norm().item()
print(f"  Sun state norm after sentence 2: {sun_norm_s2:.4f}")
print(f"  Sun fusions so far:              {memory.sun_state.age}")

# ── Resonance tests ───────────────────────────────────────────────────────────
print("\n" + "─" * 62)
print("RESONANCE SCORES (cosine similarity with Sun memory)")
print("─" * 62)

dog_vec    = embed("dog")
purple_vec = embed("purple")

res_dog    = memory.get_sun_resonance(dog_vec)
res_purple = memory.get_sun_resonance(purple_vec)

print(f"  'dog'    resonance with Sun: {res_dog:.4f}  (high — appeared in both sentences)")
print(f"  'purple' resonance with Sun: {res_purple:.4f}  (low  — never mentioned)")

# ── Gravity gate per token ────────────────────────────────────────────────────
print("\n" + "─" * 62)
print("GRAVITY GATE VALUES (with Sun State resonance boost)")
print("─" * 62)
print(f"  {'Token':<10} {'POS':<6} {'Mass':>6}  {'Gate':>8}")
print("  " + "─" * 40)

demo_tokens = [
    ("John",  "SUBJ"),
    ("the",   "DET"),
    ("cat",   "SUBJ"),
    ("dog",   "OBJ"),
    ("it",    "SUBJ"),
    ("told",  "VERB"),
    ("because","CONJ"),
]
for word, pos_type in demo_tokens:
    vec = embed(word).to(DEVICE).float()
    gate_val = gravity_gate(vec, pos_type, memory.sun_state)
    mass = POS_MASS.get(pos_type, 0.1)
    note = " ← ejected" if gate_val < 0.1 else (" ← kept" if gate_val > 0.4 else "")
    print(f"  {word:<10} {pos_type:<6} {mass:>6.2f}  {gate_val:>8.4f}{note}")

# ── Sub-planet description for "dog" ─────────────────────────────────────────
print("\n" + "─" * 62)
print("SUB-PLANET STATE for 'dog'")
print("─" * 62)
print(f"  {dog_sub.describe()}")
print(f"  Sub-planet vector norm: {dog_sub.to_vector().norm().item():.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("SUMMARY")
print("=" * 62)
print(f"  1. Sun norm after sentence 1: {sun_norm_s1:.4f}")
print(f"  2. Sun norm after sentence 2: {sun_norm_s2:.4f}")
print(f"  3. 'dog' resonance with Sun:  {res_dog:.4f}  (should be high)")
print(f"  4. 'purple' resonance:        {res_purple:.4f}  (should be low)")
print(f"  5. Gravity gate — 'the':      {gravity_gate(embed('the').to(DEVICE).float(), 'DET', memory.sun_state):.4f}  (low mass → ejected)")
print(f"     Gravity gate — 'John':     {gravity_gate(embed('John').to(DEVICE).float(), 'SUBJ', memory.sun_state):.4f}  (high mass → kept)")
print(f"  6. Sub-planet 'dog': {dog_sub.describe()}")
print("=" * 62)
