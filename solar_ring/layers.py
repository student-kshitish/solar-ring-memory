"""SolarRingLayer: one processing layer of the Solar Ring Memory model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    D_MODEL, MAX_RINGS, SLOTS_PER_RING, FLAT_SIZE,
    NUM_ROLES, CROSS_RING_LAYER, PRONOUN_LAYER, RELATION_LAYER,
    ROLE_SUBJ, ROLE_OBJ, ROLE_VERB, ROLE_CONJ
)
from .solar_memory import SolarMemory


class SolarRingLayer(nn.Module):
    """
    One stacked layer of Solar Ring Memory processing.

    Per-token operations (in order):
      1. POS classifier  → predicted role
      2. Spawn gate       → whether to spawn child ring
      3. Subject gate     → write-once subject (hard lock)
      4. Object gate      → write-once object (hard lock)
      5. Verb gate        → gated LSTM-style verb update
      6. Rotating write   → circular buffer for other roles
      7. (layer 5 only)   Cross-ring attention
      8. (layer 6 only)   Pronoun resolution
      9. (layer 7 only)   Relation encoder
     10. Output gate + residual LayerNorm

    layer_idx: 0-indexed layer number
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = D_MODEL

        # 1. POS / role classifier
        self.W_role = nn.Linear(d, NUM_ROLES)

        # 2. Spawn gate
        self.W_spawn = nn.Linear(d, 1)

        # 3. Subject projection  (write-once enforced in SolarMemory)
        self.W_subj = nn.Linear(d, d)

        # 4. Object projection
        self.W_obj = nn.Linear(d, d)

        # 5. Verb gate + projection
        self.W_verb_gate = nn.Linear(d, 1)
        self.W_verb_c = nn.Linear(d, d)

        # 6. Rotating buffer projection
        self.W_rot = nn.Linear(d, d)

        # 7. Solar Physics Attention (layer 5, idx=4) — replaces cross-ring attention
        if layer_idx == CROSS_RING_LAYER:
            from .solar_physics_attention import SolarPhysicsAttention
            self.spa = SolarPhysicsAttention(d)

        # 8. Pronoun resolution (layer 6, idx=5)
        if layer_idx == PRONOUN_LAYER:
            self.W_pro = nn.Linear(d, d)

        # 9. Relation encoder (layer 7, idx=6)
        if layer_idx == RELATION_LAYER:
            self.W_rel = nn.Linear(d * 2, d)

        # 10. Output gate + LayerNorm
        self.W_out_gate = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

        # Level 2 — multi-planet broadcast gate (created lazily on first call)
        self._W_planet_gates = None

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,          # (d,) single token embedding
        memory: SolarMemory,
        role_label: int = None,   # ground-truth role (for supervision)
        is_pronoun: bool = False,
        write_enabled: bool = True,  # only layer 0 writes to memory
    ):
        """
        Process one token x through this layer, updating memory in-place.
        write_enabled=True only for layer 0; other layers only read memory.

        Returns:
            x_out:         (d,) updated token representation
            role_logits:   (NUM_ROLES,) for POS loss
            spawn_logit:   scalar for spawn loss
        """
        d = D_MODEL
        dtype = x.dtype

        # ── 1. POS classifier ──────────────────────────────────────────
        role_logits = self.W_role(x.float()).to(dtype)          # (NUM_ROLES,)
        pred_role = role_logits.argmax().item()

        # Use ground-truth role if provided (teacher forcing), else predicted
        effective_role = role_label if role_label is not None else pred_role

        # ── 2. Spawn gate ──────────────────────────────────────────────
        spawn_logit = self.W_spawn(x.float()).squeeze(-1).to(dtype)  # scalar
        spawn_prob = torch.sigmoid(spawn_logit)
        do_spawn = (spawn_prob > 0.5).item() or (effective_role == ROLE_CONJ)

        # ── 3-6. Write to memory ring (layer 0 only) ───────────────────
        if write_enabled:
            if effective_role == ROLE_SUBJ:
                vec = self.W_subj(x.float()).to(dtype)
                memory.process_token(vec, ROLE_SUBJ, spawn=False)

            elif effective_role == ROLE_OBJ:
                vec = self.W_obj(x.float()).to(dtype)
                memory.process_token(vec, ROLE_OBJ, spawn=False)

            elif effective_role == ROLE_VERB:
                gate = torch.sigmoid(self.W_verb_gate(x.float())).squeeze(-1).to(dtype)
                vec = self.W_verb_c(x.float()).to(dtype)
                memory.active_ring.write_verb(vec, gate)

            else:
                # Rotating buffer (also handles CONJ after potential spawn)
                if effective_role == ROLE_CONJ and do_spawn:
                    memory.process_token(x, ROLE_CONJ, spawn=True)
                else:
                    vec = self.W_rot(x.float()).to(dtype)
                    memory.active_ring.write_rotating(vec)

        # ── 7. Solar Physics Attention (layer 5 only) ──────────────────
        x_ctx = x
        if self.layer_idx == CROSS_RING_LAYER:
            from .solar_physics_attention import OrbitalConcept
            _ROLE_TO_POS = {
                1: "SUBJ", 2: "OBJ", 3: "VERB", 4: "PREP",
                5: "CONJ", 6: "ADJ", 7: "ADV",  8: "DET", 0: "DET",
            }

            # Build one OrbitalConcept per ring in memory
            ring_concepts  = []
            ring_vecs_list = []
            for ring in memory.rings:
                sv  = ring.summary_vector().float()
                dep = ring.depth if ring.depth is not None else 0
                pos = "SUBJ" if ring.subj_locked else ("OBJ" if ring.obj_locked else "VERB")
                conf = 0.9 if (ring.subj_locked or ring.obj_locked) else 0.5
                ring_concepts.append(OrbitalConcept(sv, pos, dep, conf, sv.device, token_text=""))
                ring_vecs_list.append(sv)

            # Current-token concept
            cur_pos  = _ROLE_TO_POS.get(int(effective_role), "DET")
            cur_dep  = memory.active_ring.depth if memory.active_ring.depth is not None else 0
            cur_conf = float(spawn_prob.detach().item())
            x_concept = OrbitalConcept(x.float(), cur_pos, cur_dep, cur_conf, x.device, token_text="")

            # Assemble: rings first, current token last
            all_concepts = ring_concepts + [x_concept]
            ring_t   = torch.stack(ring_vecs_list, dim=0) if ring_vecs_list \
                       else x.float().unsqueeze(0)
            all_vecs = torch.cat([ring_t, x.float().unsqueeze(0)], dim=0)  # (N+1, d)

            out, _, _ = self.spa(all_concepts, all_vecs)   # (N+1, d)
            x_ctx = out[-1].to(dtype)                       # current token output

        # ── 8. Pronoun resolution (layer 6 only) ───────────────────────
        if self.layer_idx == PRONOUN_LAYER and is_pronoun:
            resolved = memory.resolve_pronoun(x_ctx)          # (d,)
            x_ctx = self.W_pro(resolved.float()).to(dtype)

        # ── 9. Relation encoder (layer 7 only) ─────────────────────────
        if self.layer_idx == RELATION_LAYER:
            ring = memory.active_ring
            subj = ring.subject_vector()
            obj  = ring.object_vector()
            pair = torch.cat([subj.float(), obj.float()], dim=-1)  # (2d,)
            rel  = torch.tanh(self.W_rel(pair)).to(dtype)          # (d,)
            x_ctx = x_ctx + rel

        # ── 10. Output gate + residual LayerNorm ───────────────────────
        gate_out = torch.sigmoid(self.W_out_gate(x_ctx.float())).to(dtype)
        x_out = self.norm((x + gate_out * x_ctx).float()).to(dtype)

        return x_out, role_logits, spawn_logit

    def forward_with_physics(self, x_t, memory,
                             token_text='', token_pos=0,
                             manager=None, sun_state=None):
        """
        Full physics-enhanced forward pass.
        Calls existing forward() first then adds:
        1. Black/white hole event checking
        2. Solar Spring attention on top of output

        Returns same signature as forward() so it is
        a drop-in enhancement not a replacement.
        """
        # Step 1: run existing forward pass unchanged
        h_t, r_t, spawn_prob = self.forward(
            x_t, memory,
            write_enabled=(self.layer_idx == 0),
        )

        # Step 2: black/white hole events
        if manager is not None and sun_state is not None:
            events = manager.step(token_text, memory, sun_state)
            if events:
                pass  # events logged — rings updated in-place

        # Step 3: solar spring enhancement on layer 5 only (CROSS_RING_LAYER=4)
        if self.layer_idx == CROSS_RING_LAYER:
            if not hasattr(self, 'spring_attn'):
                from solar_ring.solar_spring import SolarSpringAttention
                self.spring_attn = SolarSpringAttention(
                    D_MODEL
                ).to(x_t.device)

            # Build concept list from memory rings using RingNode API
            concepts = []
            vecs_list = []
            for i, ring in enumerate(memory.rings):
                depth = ring.depth if ring.depth is not None else 0
                concepts.append({
                    'pos_idx':   depth,
                    'depth':     depth,
                    'token_pos': token_pos,
                    'slot_idx':  i,
                })
                vec = ring.subject_vector() if ring.subj_locked else h_t
                vecs_list.append(vec.float())

            if len(concepts) > 1:
                vecs_t = torch.stack(vecs_list)
                spring_out, _, _ = self.spring_attn(
                    concepts, vecs_t
                )
                # Residual: add spring output to h_t
                h_t = h_t + (0.1 * spring_out.mean(0)).to(h_t.dtype)

        return h_t, r_t, spawn_prob

    def parallel_planet_broadcast(self, x_t: torch.Tensor, memory) -> torch.Tensor:
        """
        Level 2 — multi-planet parallelism.

        Broadcast x_t simultaneously to all planet rings (depth=1).
        Gate computation is a single batched Linear instead of P sequential
        calls — theoretical speedup ~P on GPU.

        Returns gate vector (P,) or None if no planets exist.
        """
        planet_rings = [r for r in memory.rings if r.depth == 1]
        if not planet_rings:
            return None

        P = len(planet_rings)
        d = x_t.shape[0]

        # Lazy-init the gate projection (P varies; recreate if size changed)
        if self._W_planet_gates is None or self._W_planet_gates.out_features != P:
            self._W_planet_gates = nn.Linear(2 * d, P).to(x_t.device)

        # Stack planet subject vectors → mean context
        planet_heads = torch.stack(
            [r.subject_vector().float() for r in planet_rings]
        )                                                 # (P, d)
        planet_mean  = planet_heads.mean(dim=0)           # (d,)
        gate_input   = torch.cat([x_t.float(), planet_mean])  # (2d,)

        # One batched gate computation for all P planets
        gates     = torch.sigmoid(self._W_planet_gates(gate_input))  # (P,)
        candidate = torch.tanh(x_t)

        for i, ring in enumerate(planet_rings):
            if gates[i].item() > 0.3:
                ring.write_rotating(candidate * gates[i])

        return gates
