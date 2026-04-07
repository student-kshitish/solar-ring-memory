"""
Demo showing Sub-Galaxy and Multiverse parallelism.
"""

import torch
import sys
sys.path.insert(0, '.')

from solar_ring.solar_memory import SolarMemory
from solar_ring.sun_state import SunState
from solar_ring.multi_solar_system import MultiSolarSystem
from solar_ring.galaxy import (
    GalacticCore, SubGalaxy, MultiverseMemory
)

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')
D = 300

def demo_multiverse():
    print("="*60)
    print("MULTIVERSE DEMO")
    print("Ambiguous sentence: 'The bank was steep.'")
    print("="*60)

    core = GalacticCore(D, DEVICE)
    multiverse = MultiverseMemory(D, DEVICE, core)
    base_mem = SolarMemory(device=DEVICE)

    tokens = [
        ("The", "DET"),
        ("bank", "SUBJ"),    # <- ambiguous!
        ("was", "VERB"),
        ("steep", "ADJ"),
        (".",   "OTHER"),
    ]

    print("\nToken processing:")
    for word, pos in tokens:
        spawned = multiverse.spawn_universes(word, base_mem)
        if not multiverse.is_collapsed:
            ctx_vec = torch.randn(D, device=DEVICE)
            multiverse.update_probabilities(ctx_vec)
            print(f"  {word:10} {pos:6} "
                  f"universes={multiverse.parallel_step_count()} "
                  f"P={[f'{p:.2f}' for p in multiverse.probabilities]}")
        else:
            print(f"  {word:10} {pos:6} single universe")

    active = multiverse.get_active_memory()
    print(f"\nActive universe: {multiverse.labels[0] if not multiverse.is_collapsed else 'collapsed'}")

def demo_sub_galaxy():
    print("\n" + "="*60)
    print("SUB-GALAXY DEMO")
    print("Multiple documents about same topic clustering")
    print("="*60)

    core = GalacticCore(D, DEVICE, beta=0.1)
    cluster = SubGalaxy(D, DEVICE, core)

    docs = [
        "John studied ecology near the river bank.",
        "The steep river bank eroded during floods.",
        "Mary measured the bank's gradient carefully.",
    ]

    print("\nAdding documents to sub-galaxy:")
    for i, doc in enumerate(docs):
        mss = MultiSolarSystem(D, DEVICE)

        # Simulate processing
        vec = torch.randn(D, device=DEVICE)
        mss.active_sun.fuse([vec])
        mss.end_paragraph()

        cluster.add_document(mss)
        sim = core.topic_similarity(
            mss.active_sun.state
        )
        print(f"  Doc {i+1}: similarity={sim:.3f} "
              f"centroid_norm={cluster.centroid.norm():.3f}")

    print(f"\nGalactic core norm: {core.state.norm():.3f}")
    print(f"Documents in cluster: {len(cluster.systems)}")

    # Cross-document query
    query = torch.randn(D, device=DEVICE)
    result = cluster.inter_cluster_gravity(query, top_k=2)
    print(f"Inter-cluster gravity result norm: {result.norm():.3f}")
    print("Most relevant documents found via sub-galaxy gravity.")

if __name__ == "__main__":
    demo_multiverse()
    demo_sub_galaxy()

    print("\n" + "="*60)
    print("HIERARCHY SUMMARY")
    print("="*60)
    print("Multiverse  -> parallel ambiguous interpretations")
    print("Galaxy      -> all documents on same topic")
    print("Sub-galaxy  -> clusters of related paragraphs")
    print("Solar System-> one document")
    print("Planet      -> one clause ring")
    print("Sub-planet  -> animacy/case/size slots")
    print("="*60)
