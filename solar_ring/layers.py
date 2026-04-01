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

        # 7. Cross-ring attention (layer 5, idx=4)
        if layer_idx == CROSS_RING_LAYER:
            self.W_Q = nn.Linear(d, d)
            self.W_K = nn.Linear(d, d)
            self.W_V = nn.Linear(d, d)
            self.W_cross_out = nn.Linear(d, d)

        # 8. Pronoun resolution (layer 6, idx=5)
        if layer_idx == PRONOUN_LAYER:
            self.W_pro = nn.Linear(d, d)

        # 9. Relation encoder (layer 7, idx=6)
        if layer_idx == RELATION_LAYER:
            self.W_rel = nn.Linear(d * 2, d)

        # 10. Output gate + LayerNorm
        self.W_out_gate = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

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

        # ── 7. Cross-ring attention (layer 5 only) ─────────────────────
        x_ctx = x
        if self.layer_idx == CROSS_RING_LAYER:
            summaries = memory.get_summary_vectors()          # (13, d)
            Q = self.W_Q(x.float().unsqueeze(0)).to(dtype)   # (1, d)
            K = self.W_K(summaries.float()).to(dtype)         # (13, d)
            V = self.W_V(summaries.float()).to(dtype)         # (13, d)
            scale = d ** 0.5
            attn = F.softmax((Q @ K.T) / scale, dim=-1)      # (1, 13)
            x_ctx = (attn @ V).squeeze(0)                    # (d,)
            x_ctx = self.W_cross_out(x_ctx.float()).to(dtype)

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
