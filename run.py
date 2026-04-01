#!/usr/bin/env python3
"""
run.py — Solar Ring Memory entry point.

Checks GPU, installs deps if missing, trains for 1 epoch on sample
sentences, then runs the demo sentence and prints ring contents.
"""

import sys
import subprocess


# ── 0. Dependency check / install ─────────────────────────────────────────────

def ensure_deps():
    required = ["torch", "spacy"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

    # spaCy model
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model en_core_web_sm...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True
        )


ensure_deps()

# ── 1. Imports ────────────────────────────────────────────────────────────────

import torch
from solar_ring.train import train, SAMPLE_SENTENCES
from solar_ring.dataset import SolarRingDataset
from solar_ring.model import SolarRingModel
from solar_ring.pos_tagger import POSTagger

# ── 2. GPU check ──────────────────────────────────────────────────────────────

def check_device():
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        mem   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {name} ({mem:.1f} GB VRAM)")
        return torch.device("cuda")
    else:
        print("No GPU detected, using CPU.")
        return torch.device("cpu")


# ── 3. Demo: inspect ring contents ───────────────────────────────────────────

DEMO_SENTENCE = "John told Mary that the cat chased the dog because it was too big"

def run_demo(model: SolarRingModel, dataset: SolarRingDataset, device):
    print("\n" + "="*70)
    print("DEMO SENTENCE:")
    print(f"  {DEMO_SENTENCE}")
    print("="*70)

    tagger = POSTagger()
    tags   = tagger.tag(DEMO_SENTENCE)

    # Build token_ids / role_labels / spawn_labels / pronoun_mask
    from solar_ring.dataset import PRONOUNS
    from solar_ring.config import ROLE_OTHER

    tokens = [t["text"] for t in tags]
    roles  = [t["role"] for t in tags]
    spawns = [1.0 if t["spawn"] else 0.0 for t in tags]
    is_pron= [t["text"].lower() in PRONOUNS for t in tags]

    w2id = dataset.word2id
    ids  = (
        [w2id["<BOS>"]]
        + [w2id.get(w.lower(), w2id["<UNK>"]) for w in tokens]
        + [w2id["<EOS>"]]
    )
    roles   = [ROLE_OTHER] + roles   + [ROLE_OTHER]
    spawns  = [0.0]        + spawns  + [0.0]
    is_pron = [False]      + is_pron + [False]

    token_ids    = torch.tensor(ids,     dtype=torch.long,  device=device)
    role_labels  = torch.tensor(roles,   dtype=torch.long,  device=device)
    pronoun_mask = torch.tensor(is_pron, dtype=torch.bool,  device=device)

    # Run model and capture memory
    model.eval()
    memory = model.get_memory_for_sentence(
        token_ids, role_labels=role_labels, pronoun_mask=pronoun_mask
    )

    # Print token → tag mapping
    print("\nToken analysis:")
    print(f"  {'Token':<12} {'POS':<8} {'Dep':<10} {'Role':<6} {'Spawn'}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*6} {'-'*5}")
    for t in tags:
        print(f"  {t['text']:<12} {t['pos']:<8} {t['dep']:<10} {t['role']:<6} {t['spawn']}")

    # Print ring contents
    print(f"\nRing memory ({len(memory.rings)} rings active / 13 max):")
    memory.print_rings()

    # Print flatten shape
    flat = memory.flatten()
    print(f"\nFlattened memory shape: {flat.shape}  (MAX_RINGS×SLOTS×D = 13×8×512=53248)")
    print(f"Flat norm: {flat.norm().item():.4f}")
    print("\nDone.")


# ── 4. Main ───────────────────────────────────────────────────────────────────

def main():
    device = check_device()
    print("\nSetup check passed. Starting training...\n")

    model, dataset = train(
        sentences=SAMPLE_SENTENCES,
        n_epochs=1,
        device=device,
    )

    run_demo(model, dataset, device)


if __name__ == "__main__":
    main()
