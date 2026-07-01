"""SRM -> coref bridge types and projection.

Skeleton for the option-1 seam described in `coref/wiring_notes.md`. SRM's role
is memory + merging ONLY: it hands compressed cross-window entity vectors to the
trained relational decider. Nothing here decides antecedents, and nothing here
trains.

Shapes: SRM entity/sun vectors are 300-d (D_MODEL, GloVe-300d). The coref pair
space is 256-d (span_proj_dim in coref/model.py). This module owns the 300->256
projection and the hand-off record type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

SRM_DIM: int = 300     # D_MODEL — SunState.state / UnifiedMemory entity vec
PAIR_DIM: int = 256    # span_proj_dim in coref/model.py (pair space)


class SrmAntecedentProjector(nn.Module):
    """Project a 300-d SRM entity vector into the 256-d coref pair space.

    Tiny and fully implemented (Linear + LayerNorm). Its *weights* are trained
    tomorrow on coref data alongside the pair scorer; construction here is
    laptop-safe and runs no forward pass by itself.
    """

    def __init__(self, srm_dim: int = SRM_DIM, pair_dim: int = PAIR_DIM):
        super().__init__()
        self.proj = nn.Linear(srm_dim, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)

    def forward(self, srm_vec: Tensor) -> Tensor:
        """srm_vec: [*, 300] -> [*, 256]."""
        return self.norm(self.proj(srm_vec))


@dataclass
class SrmMention:
    """One cross-window entity handed from SRM memory to the decider.

    Attributes:
        entity_id: stable id for cluster bookkeeping across chunks.
        repr_300:  [300] SRM entity/sun representation (pre-projection).
        start:     virtual document position, assigned EARLIER than any
                   current-chunk anaphor so the existing tril(diagonal=-1)
                   causal mask treats it as a valid antecedent with no edit.
        source:    which SRM store produced the vector.
    """

    entity_id: str
    repr_300: Tensor
    start: int
    source: Literal["unified_memory", "sun_state"]


def gather_srm_antecedents(memory) -> list[SrmMention]:
    """Read cross-window entities out of SRM memory into SrmMention records.

    Intended source: UnifiedMemory entity vectors (primary) or
    MultiSolarSystem sun_states / gravity_waves (coarse fallback). Left
    unimplemented tonight — this is a memory-read to be wired tomorrow.
    """
    raise NotImplementedError("wire to UnifiedMemory/SunState tomorrow")
