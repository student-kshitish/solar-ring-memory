"""SolarMemory: manages the full ring hierarchy for one sequence."""

import torch
import torch.nn.functional as F
from .config import (
    D_MODEL, MAX_RINGS, SLOTS_PER_RING, FLAT_SIZE,
    ROLE_SUBJ, ROLE_OBJ, ROLE_VERB, ROLE_CONJ
)
from .ring_node import RingNode


class SolarMemory:
    """
    Manages a list of up to MAX_RINGS RingNodes.

    Hierarchy:
        rings[0] = sun (main clause)
        rings[1..4] = planets (depth=1)
        rings[5..12] = moons (depth=2)

    alpha: index of the currently active ring.
    """

    def __init__(self, device="cpu", dtype=torch.bfloat16, hard_lock: bool = False):
        # Normalise device to torch.device so str "cuda"/"cpu" both work
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype
        self.hard_lock = hard_lock  # False=soft (training), True=hard (inference)

        # Initialise with the sun ring
        sun = RingNode(device=device, dtype=dtype, ring_id=0, parent_id=None)
        sun.depth = 0
        self.rings = [sun]
        self.alpha = 0  # active ring pointer

    # ------------------------------------------------------------------
    # Ring management
    # ------------------------------------------------------------------

    def _spawn(self, parent_id: int) -> int:
        """
        Spawn a new child ring. Returns new ring id, or -1 if at capacity.
        """
        if len(self.rings) >= MAX_RINGS:
            return -1  # capacity reached, stay in current ring

        new_id = len(self.rings)
        parent = self.rings[parent_id]
        child = RingNode(
            device=self.device, dtype=self.dtype,
            ring_id=new_id, parent_id=parent_id
        )
        child.depth = parent.depth + 1
        self.rings.append(child)
        return new_id

    def activate(self, ring_id: int):
        self.alpha = ring_id

    @property
    def active_ring(self) -> RingNode:
        return self.rings[self.alpha]

    # ------------------------------------------------------------------
    # Token processing entry point
    # ------------------------------------------------------------------

    def process_token(self, vec: torch.Tensor, role_id: int,
                      verb_gate: torch.Tensor = None,
                      spawn: bool = False,
                      token_text: str = "") -> int:
        """
        Route vec into the active ring based on role_id.
        If spawn=True, create a child ring and switch to it first.
        Returns the active ring index after processing.
        """
        if spawn and role_id == ROLE_CONJ:
            new_id = self._spawn(self.alpha)
            if new_id != -1:
                self.alpha = new_id

        ring = self.active_ring

        if role_id == ROLE_SUBJ:
            ring.write_subject(vec, hard_lock=self.hard_lock)
            if token_text:
                ring.subj_word = token_text
        elif role_id == ROLE_OBJ:
            ring.write_object(vec, hard_lock=self.hard_lock)
            if token_text:
                ring.obj_word = token_text
        elif role_id == ROLE_VERB:
            gate = verb_gate if verb_gate is not None else torch.tensor(0.5, dtype=self.dtype, device=self.device)
            ring.write_verb(vec, gate)
        else:
            ring.write_rotating(vec)

        return self.alpha

    # ------------------------------------------------------------------
    # Pronoun resolution
    # ------------------------------------------------------------------

    def resolve_pronoun(self, x: torch.Tensor) -> torch.Tensor:
        """
        Walk from alpha toward sun (ring 0), scoring each ring's
        subject slot. Return softmax-weighted sum of candidates.
        x: (D_MODEL,) query vector
        """
        # Collect ancestors including current
        path = []
        cur = self.alpha
        while cur is not None:
            path.append(cur)
            cur = self.rings[cur].parent_id

        if not path:
            return x

        # Gather subject vectors as candidates
        candidates = []
        for rid in path:
            subj = self.rings[rid].subject_vector()
            candidates.append(subj)

        # Stack and score
        C = torch.stack(candidates, dim=0)  # (k, D_MODEL)
        scale = D_MODEL ** 0.5
        scores = (x @ C.T) / scale          # (k,)
        weights = F.softmax(scores, dim=0)  # (k,)
        resolved = (weights.unsqueeze(-1) * C).sum(dim=0)  # (D_MODEL,)
        return resolved

    # ------------------------------------------------------------------
    # Flatten to fixed size
    # ------------------------------------------------------------------

    def flatten(self) -> torch.Tensor:
        """
        Return always-fixed (MAX_RINGS * SLOTS_PER_RING * D_MODEL,) tensor.
        Pads with zeros for unused rings.
        """
        parts = []
        for i in range(MAX_RINGS):
            if i < len(self.rings):
                parts.append(self.rings[i].to_vector())  # (8, D_MODEL)
            else:
                parts.append(torch.zeros(SLOTS_PER_RING, D_MODEL,
                                         device=self.device, dtype=self.dtype))  # padded on device
        stacked = torch.stack(parts, dim=0)   # (13, 8, D_MODEL)
        return stacked.reshape(-1)            # (53248,)

    def get_summary_vectors(self) -> torch.Tensor:
        """
        Return (MAX_RINGS, D_MODEL) summary vectors (one per ring).
        Used for cross-ring attention.
        """
        parts = []
        for i in range(MAX_RINGS):
            if i < len(self.rings):
                parts.append(self.rings[i].summary_vector())
            else:
                parts.append(torch.zeros(D_MODEL, device=self.device, dtype=self.dtype))  # on device
        return torch.stack(parts, dim=0)  # (13, D_MODEL)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self):
        sun = RingNode(device=self.device, dtype=self.dtype, ring_id=0, parent_id=None)
        sun.depth = 0
        self.rings = [sun]
        self.alpha = 0

    def __len__(self):
        return len(self.rings)

    def __repr__(self):
        return (f"SolarMemory(rings={len(self.rings)}/{MAX_RINGS}, "
                f"alpha={self.alpha})")

    def print_rings(self):
        """Pretty-print ring contents for debugging."""
        labels = ["SUBJ", "OBJ ", "VERB", "ROT0", "ROT1", "ROT2", "ROT3", "ROT4"]
        depth_names = {0: "SUN", 1: "PLANET", 2: "MOON"}
        for ring in self.rings:
            dn = depth_names.get(ring.depth, f"DEPTH{ring.depth}")
            print(f"\n--- Ring {ring.ring_id} [{dn}] parent={ring.parent_id} ---")
            for i in range(8):
                norm = ring.slot_norm(i)
                locked = ""
                if i == 0:
                    locked = " [LOCKED]" if ring.subj_locked else " [free]"
                elif i == 1:
                    locked = " [LOCKED]" if ring.obj_locked else " [free]"
                print(f"  slot[{i}] {labels[i]}: norm={norm:.4f}{locked}")
