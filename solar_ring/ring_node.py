"""RingNode: 8-slot memory ring for one clause."""

import torch
from .config import (
    D_MODEL, SLOTS_PER_RING, SUBJ_SLOT, OBJ_SLOT, VERB_SLOT,
    ROT_START, ROTATING_SLOTS
)


class RingNode:
    """
    8-slot memory ring for a single clause.

    Slots:
        0 - SUBJ: write-once lock (hard guarantee)
        1 - OBJ:  write-once lock (hard guarantee)
        2 - VERB: gated update
        3-7 - rotating circular buffer (5 slots)

    Each slot is stored as an independent tensor to avoid in-place
    autograd conflicts. Slots start as None (zero vector on read).
    """

    def __init__(self, device="cpu", dtype=torch.bfloat16, ring_id=0, parent_id=None):
        self.device = device
        self.dtype = dtype
        self.ring_id = ring_id
        self.parent_id = parent_id

        # Pre-allocate zero tensors on the target device so no lazy CPU fallback
        self._zero_vec = torch.zeros(D_MODEL, device=device, dtype=dtype)

        # Store each slot independently (avoids in-place modification issues)
        # None means "not yet written" → reads as zeros
        self._slots: list = [None] * SLOTS_PER_RING  # list of tensors or None

        # Write-once locks for subject and object
        self.subj_locked = False
        self.obj_locked = False

        # Circular buffer head pointer for rotating slots (0..4)
        self.rot_head = 0

        # Depth in the tree (0 = sun, 1 = planet, 2 = moon)
        self.depth = 0 if parent_id is None else None

    def _zero(self) -> torch.Tensor:
        # Return a fresh zero tensor on the correct device (never fallback to CPU)
        return torch.zeros(D_MODEL, device=self.device, dtype=self.dtype)

    def _get(self, idx: int) -> torch.Tensor:
        """Return slot tensor, or zero if never written."""
        s = self._slots[idx]
        return s if s is not None else self._zero()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_subject(self, vec: torch.Tensor, hard_lock: bool = False) -> bool:
        """
        Write subject slot. Returns True if the slot was updated.

        hard_lock=True  (inference): original write-once guarantee — first
            write wins, all subsequent writes are silently dropped.
        hard_lock=False (training):  soft pole — the first write sets the slot
            normally; subsequent writes are blended using a forget gate derived
            from the existing content's norm.  A large norm (well-established
            pole) produces a near-zero forget gate so the pole is effectively
            frozen; a small norm (early training) allows more refinement.
            Gradient flows through the new write path, enabling the model to
            correct early wrong assignments during training.
        """
        if hard_lock:
            if self.subj_locked:
                return False
            self._slots[SUBJ_SLOT] = vec.to(self.dtype)
            self.subj_locked = True
            return True

        s = self._slots[SUBJ_SLOT]
        if s is None:
            self._slots[SUBJ_SLOT] = vec.to(self.dtype)
            self.subj_locked = True
        else:
            # soft_lock → 1 when norm is large (pole established → resist change)
            # soft_lock → 0 when norm is small (pole uncertain → allow update)
            soft_lock   = torch.sigmoid(s.norm() * 3.0)
            forget_gate = 1.0 - soft_lock
            self._slots[SUBJ_SLOT] = (
                soft_lock * s.detach() + forget_gate * vec
            ).to(self.dtype)
        return True

    def write_object(self, vec: torch.Tensor, hard_lock: bool = False) -> bool:
        """
        Write object slot. Same soft/hard locking semantics as write_subject.
        """
        if hard_lock:
            if self.obj_locked:
                return False
            self._slots[OBJ_SLOT] = vec.to(self.dtype)
            self.obj_locked = True
            return True

        s = self._slots[OBJ_SLOT]
        if s is None:
            self._slots[OBJ_SLOT] = vec.to(self.dtype)
            self.obj_locked = True
        else:
            soft_lock   = torch.sigmoid(s.norm() * 3.0)
            forget_gate = 1.0 - soft_lock
            self._slots[OBJ_SLOT] = (
                soft_lock * s.detach() + forget_gate * vec
            ).to(self.dtype)
        return True

    def write_verb(self, vec: torch.Tensor, gate: torch.Tensor):
        """Gated verb update: v = (1-gate)*v + gate*tanh(vec)."""
        g = gate.to(self.dtype).clamp(0.0, 1.0)
        old_v = self._get(VERB_SLOT)
        # Use .detach() on old_v to avoid backpropagating through old state
        # (this is the standard LSTM-style truncated BPTT approach)
        self._slots[VERB_SLOT] = (1.0 - g) * old_v.detach() + g * torch.tanh(vec.to(self.dtype))

    def write_rotating(self, vec: torch.Tensor):
        """Write to circular buffer at current head, then advance."""
        slot_idx = ROT_START + self.rot_head
        self._slots[slot_idx] = torch.tanh(vec.to(self.dtype))
        self.rot_head = (self.rot_head + 1) % ROTATING_SLOTS

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def to_vector(self) -> torch.Tensor:
        """Return all 8 slots as (8, D_MODEL) tensor (stacked)."""
        return torch.stack([self._get(i) for i in range(SLOTS_PER_RING)], dim=0)

    def summary_vector(self) -> torch.Tensor:
        """Mean of all written slots as a single D_MODEL vector."""
        written = [self._slots[i] for i in range(SLOTS_PER_RING) if self._slots[i] is not None]
        if not written:
            return self._zero()
        return torch.stack(written, dim=0).mean(dim=0)

    def subject_vector(self) -> torch.Tensor:
        return self._get(SUBJ_SLOT)

    def object_vector(self) -> torch.Tensor:
        return self._get(OBJ_SLOT)

    def verb_vector(self) -> torch.Tensor:
        return self._get(VERB_SLOT)

    def slot_norm(self, idx: int) -> float:
        s = self._slots[idx]
        return s.detach().norm().item() if s is not None else 0.0

    def __repr__(self):
        locked = f"subj={'locked' if self.subj_locked else 'free'}, obj={'locked' if self.obj_locked else 'free'}"
        return f"RingNode(id={self.ring_id}, parent={self.parent_id}, {locked}, rot_head={self.rot_head})"
