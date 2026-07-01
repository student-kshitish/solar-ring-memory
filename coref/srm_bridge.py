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
        entity_id:  stable id for cluster bookkeeping across chunks.
        repr_300:   [300] SRM entity/sun representation (pre-projection).
        start:      virtual document position, assigned EARLIER than any
                    current-chunk anaphor so the existing tril(diagonal=-1)
                    causal mask treats it as a valid antecedent with no edit.
        source:     which SRM store produced the vector.
        confidence: merge confidence carried from the merging layer; seeds the
                    antecedent's mention-score in prepend_to_pairwise. A proxy
                    (entity mass) until the merger emits a real confidence.
    """

    entity_id: str
    repr_300: Tensor
    start: int
    source: Literal["unified_memory", "sun_state"]
    confidence: float = 0.0


def gather_srm_antecedents(memory, min_anaphor_start: int = 0) -> list[SrmMention]:
    """Read cross-window entities out of UnifiedMemory into SrmMention records.

    Pure dict/list iteration over `memory.entities` — no torch model, no forward
    pass, CPU-safe. Each live entity becomes a virtual antecedent whose `start`
    is placed strictly below `min_anaphor_start` (descending: -1, -2, ... below
    the chunk's earliest anaphor), so virtual antecedents always precede
    current-chunk mentions and the existing causal mask needs no edit.

    Confidence is read from entity['confidence'] if present, else falls back to
    entity['mass'] as a stand-in proxy until the merging layer emits a real
    merge confidence tomorrow.

    Empty-safe: returns [] if memory has no entities.

    NOTE (untrained plumbing): the returned vectors are genuine SRM entity
    reprs, but their downstream use as antecedents is only meaningful after the
    projector + pair head are trained (step 3). Shapes are correct; scores built
    on top are noise until then.

    Args:
        memory:            a UnifiedMemory-like object exposing `.entities`
                           (dict name -> {'vec','name','mass','alive',...}).
        min_anaphor_start: the earliest word-start among current-chunk anaphors;
                           virtual starts are assigned below it. Default 0 => all
                           virtual starts are negative (precede any real token).

    Returns:
        list[SrmMention], one per live entity (empty list if none).
    """
    entities = getattr(memory, "entities", None)
    if not entities:
        return []

    mentions: list[SrmMention] = []
    for key, e in entities.items():
        if not e.get("alive", True):
            continue
        vec = e.get("vec")
        if vec is None:
            continue
        confidence = float(e.get("confidence", e.get("mass", 0.0)))
        # descending starts, each strictly below min_anaphor_start
        start = min_anaphor_start - (len(mentions) + 1)
        mentions.append(
            SrmMention(
                entity_id=e.get("name", key),
                repr_300=vec,
                start=start,
                source="unified_memory",
                confidence=confidence,
            )
        )
    return mentions
