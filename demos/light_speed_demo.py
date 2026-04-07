"""
Demo showing speed-of-light memory constraints.
"""
import torch
import sys
sys.path.insert(0, '.')

from solar_ring.light_speed import (
    LightSpeedMemory, compute_orbital_redshift,
    PARTICLE_MASS, POS_NAMES
)

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
D = 300

def demo():
    print("="*60)
    print("Speed of Light Memory Demo")
    print("="*60)

    model = LightSpeedMemory(D).to(DEVICE)

    print(f"\nSpeed of light c = {model.c_memory:.1f} tokens")
    print(f"\nParticle velocities:")
    for i, pos in enumerate(POS_NAMES):
        v = model.particle_velocity(i)
        mass = model.get_mass(i)
        print(f"  {pos:6}: mass={mass:.2f}  v={v:.1f} tok/step  "
              f"{'PHOTON' if mass < 0.01 else ''}")

    # Sentence: "John told Mary that the cat chased the dog it was hungry"
    words = ['John','told','Mary','that','the',
             'cat','chased','the','dog','it','was','hungry']

    # POS indices: SUBJ=0, VERB=1, OBJ=2, CONJ=4, DET=7, PRON=8
    pos_map = {
        'John':0,'Mary':2,'cat':0,'dog':2,
        'told':1,'chased':1,'was':1,
        'that':4,'but':4,
        'the':7,'a':7,
        'it':8,'he':8,'she':8,
        'hungry':5,'angry':5,
    }

    pos_indices = [pos_map.get(w.lower(), 9) for w in words]
    positions   = list(range(len(words)))

    token_vecs = torch.randn(
        len(words), D, device=DEVICE
    )

    print(f"\nSentence: {' '.join(words)}")
    print(f"\nCausal mask — can 'it' (pos=9) see each token?")
    pronoun_pos = 9  # position of 'it'

    for j, (word, pos_idx) in enumerate(
        zip(words, pos_indices)
    ):
        if j >= pronoun_pos:
            continue
        d = pronoun_pos - j
        cm = model.causal_mask(pronoun_pos, j, pos_idx)
        rs = model.redshift(d, pos_idx)
        pos_name = POS_NAMES[pos_idx % len(POS_NAMES)]
        status = 'VISIBLE' if cm > 0 else 'BLOCKED'
        print(f"  {word:8} pos={j:2} dist={d:2} "
              f"type={pos_name:6} "
              f"causal={status:7} redshift={rs:.3f}")

    print(f"\nLight cone analysis:")
    print(f"  c_memory = {model.c_memory:.1f} tokens")
    print(f"  'it' (photon, mass=0.0) sees ALL past tokens")
    print(f"  'John' (noun, mass=0.95) travels at "
          f"v={model.particle_velocity(0):.1f} tok/step")
    print(f"  'the' (det, mass=0.05) travels at "
          f"v={model.particle_velocity(7):.1f} tok/step")

    print(f"\nRedshift table:")
    print(f"  {'Distance':>10} {'Noun λ':>10} "
          f"{'Verb λ':>10} {'Photon λ':>10}")
    for d in [1, 5, 10, 25, 50, 100, 200]:
        lam_noun  = model.redshift(d, 0)  # SUBJ
        lam_verb  = model.redshift(d, 1)  # VERB
        lam_phot  = model.redshift(d, 8)  # PRON (photon)
        print(f"  {d:>10} {lam_noun:>10.3f} "
              f"{lam_verb:>10.3f} {lam_phot:>10.3f}")

    print(f"\nOrbital redshift between ring depths:")
    for d1, d2 in [(0,0),(0,1),(0,2),(1,2)]:
        rs = compute_orbital_redshift(d1, d2)
        print(f"  depth {d1} → depth {d2}: λ={rs:.3f}")

    # Full forward pass
    attended, scores, photon_mask = model(
        token_vecs, positions, pos_indices
    )

    print(f"\nForward pass output:")
    print(f"  attended shape: {attended.shape}")
    print(f"  Photon tokens: "
          f"{[words[i] for i,p in enumerate(photon_mask) if p>0.5]}")
    print(f"  Non-zero attention pairs: "
          f"{(scores > 0).sum().item()}/{scores.numel()}")

    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("Transformers treat all tokens equally (no redshift penalty)")
    print("Solar Ring+Light: token 99 noun arrives at 37% strength")
    print("  after 100 tokens (redshift λ=e^(-100/50)=0.135)")
    print("  But pronoun 'it' sees full past (photon, λ=1.0)")
    print("This enforces linguistic causality transformers ignore.")

if __name__ == "__main__":
    demo()
