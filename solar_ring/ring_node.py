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
        self.obj_locked  = False
        # Text label of word written to subject/object slot (knowledge injection)
        self.subj_word: str = ""
        self.obj_word:  str = ""

        # Circular buffer head pointer for rotating slots (0..4)
        self.rot_head = 0

        # Depth in the tree (0 = sun, 1 = planet, 2 = moon)
        self.depth = 0 if parent_id is None else None

        # Named moon dict — parallel interface for the 5 rotating slots
        self.moons: dict = {f"moon_{i}": self._zero() for i in range(ROTATING_SLOTS)}

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

    def write_all_moons_parallel(
        self,
        token_vec: torch.Tensor,
        relevance_gates: dict,
    ) -> None:
        """
        Update all moon slots simultaneously using a single matrix operation.

        relevance_gates: dict of {moon_name: gate_value} — one gate per moon.
        All moons update in one batched operation instead of a sequential loop.

        Old (sequential): for moon in moons: moon.update(token_vec)
        New (parallel):   one CUDA kernel — all moons updated simultaneously.
        """
        M = len(self.moons)
        moon_names = list(self.moons.keys())

        moon_matrix = torch.stack([
            self.moons[n] for n in moon_names
        ])  # (M, d)

        gate_vec = torch.tensor(
            [relevance_gates.get(n, 0.0) for n in moon_names],
            device=token_vec.device,
            dtype=token_vec.dtype,
        )  # (M,)

        candidate = torch.tanh(token_vec).unsqueeze(0).expand(M, -1)  # (M, d)

        updated = (
            (1 - gate_vec.unsqueeze(1)) * moon_matrix +
            gate_vec.unsqueeze(1) * candidate
        )  # (M, d) — one operation

        for i, name in enumerate(moon_names):
            self.moons[name] = updated[i]

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


# ── Sub-planet parallelism speedup demo ──────────────────────────────────────

if __name__ == "__main__":
    import time
    import torch
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    DTYPE   = torch.float32
    N_ITER  = 10_000   # warmup + benchmark iterations

    def _new_node():
        return RingNode(device=DEVICE, dtype=DTYPE, ring_id=0, parent_id=None)

    gates = {f"moon_{i}": 0.5 for i in range(ROTATING_SLOTS)}

    # Pre-allocate tensors for fair GPU timing (avoids alloc overhead in loop)
    M        = ROTATING_SLOTS       # 5 moon slots
    vecs     = torch.randn(N_ITER, D_MODEL, device=DEVICE, dtype=DTYPE)
    gate_t   = torch.full((M,), 0.5, device=DEVICE, dtype=DTYPE)

    # ── Warmup ────────────────────────────────────────────────────────────────
    node_w = _new_node()
    moon_mat_w = torch.stack(list(node_w.moons.values())).to(DEVICE)
    for i in range(200):
        v = vecs[i % N_ITER]
        # sequential
        node_w.write_rotating(v)
        # parallel (kernel)
        cand       = torch.tanh(v).unsqueeze(0).expand(M, -1)
        moon_mat_w = (1 - gate_t.unsqueeze(1)) * moon_mat_w + gate_t.unsqueeze(1) * cand
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # ── Sequential baseline: loop over M slots one at a time ─────────────────
    # Simulate sequential update of each moon with its own tanh+blend
    moon_seq = torch.zeros(M, D_MODEL, device=DEVICE, dtype=DTYPE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(N_ITER):
        v = vecs[k]
        for i in range(M):                          # sequential per-moon loop
            moon_seq[i] = (1 - 0.5) * moon_seq[i] + 0.5 * torch.tanh(v)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    old_time = time.perf_counter() - t0

    # ── Parallel: single batched matrix op — all M moons at once ─────────────
    moon_par = torch.zeros(M, D_MODEL, device=DEVICE, dtype=DTYPE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(N_ITER):
        v    = vecs[k]
        cand = torch.tanh(v).unsqueeze(0).expand(M, -1)   # (M, d)
        moon_par = (1 - gate_t.unsqueeze(1)) * moon_par + gate_t.unsqueeze(1) * cand
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    new_time = time.perf_counter() - t0

    speedup = old_time / new_time if new_time > 0 else float("inf")
    print(f"\n{'─'*50}")
    print(f"Sub-planet parallelism speedup: {speedup:.1f}x")
    print(f"  Sequential (loop)  : {old_time*1000:.2f} ms  ({N_ITER} iters × {M} moons)")
    print(f"  Parallel (matrix)  : {new_time*1000:.2f} ms  ({N_ITER} iters × {M} moons)")
    print(f"  Moons per update   : {M}  |  D_MODEL: {D_MODEL}")
    print(f"{'─'*50}")
