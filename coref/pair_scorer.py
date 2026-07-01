"""Pair-scorer protocol + implementations for the option-1 decider.

The decider is where reasoning lives (an asymmetric, relational function whose
output changes when the alternatives change). SRM only supplies representations;
it never decides. See `coref/wiring_notes.md` and
`coref/reasoning_architecture.md`.

HARD CONSTRAINT: SRM antecedents are compressed vectors with NO raw text. Any
PairScorer implementation MUST support a vector-only antecedent path
(`antecedent_raw is None` => score from `antecedent_repr` alone). A scorer that
requires raw spans cannot link cross-window (SRM) antecedents.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor


@runtime_checkable
class PairScorer(Protocol):
    """Score a (anaphor, antecedent) pair. Higher = more likely coreferent.

    Args:
        anaphor_repr:    [*, 256] current-chunk mention repr (gi side).
        antecedent_repr: [*, 256] antecedent repr (gj side); may be an SRM
                         entity projected into the 256-d pair space.
        dist_feat:       [*, 20] distance-bucket embedding for the pair.
        anaphor_raw:     optional raw span handle (text/token ids), if available.
        antecedent_raw:  optional raw span handle. **None for SRM memory
                         antecedents** — implementations must handle this.

    Returns:
        [*] scalar score per pair.
    """

    def __call__(
        self,
        anaphor_repr: Tensor,
        antecedent_repr: Tensor,
        dist_feat: Tensor,
        *,
        anaphor_raw: Optional[Any] = None,
        antecedent_raw: Optional[Any] = None,
    ) -> Tensor:
        ...


class FfnPairScorer(PairScorer):
    """Today's decider: the EXISTING coref/model.py pair head made protocol-
    conformant. Reproduces `pair_ffn(cat([gi, gj, gi*gj, dist_feat]))`.

    Purely vector-based, so it already IS the vector-only antecedent path; the
    `*_raw` kwargs are ignored. Pass the existing `CorefModel.pair_ffn` module
    in — this wrapper adds no parameters and trains nothing on its own.
    """

    def __init__(self, pair_ffn: nn.Module):
        self.pair_ffn = pair_ffn

    def __call__(
        self,
        anaphor_repr: Tensor,
        antecedent_repr: Tensor,
        dist_feat: Tensor,
        *,
        anaphor_raw: Optional[Any] = None,
        antecedent_raw: Optional[Any] = None,
    ) -> Tensor:
        # Same feature construction and head as coref/model.py:pairwise_scores.
        pair_repr = torch.cat(
            [anaphor_repr, antecedent_repr, anaphor_repr * antecedent_repr, dist_feat],
            dim=-1,
        )
        return self.pair_ffn(pair_repr).squeeze(-1)


class CrossEncoderPairScorer(PairScorer):
    """Tomorrow's decider: an A100 cross-encoder that jointly encodes the two
    spans. STUB — signature and contract only, no implementation tonight.

    Vector-only path (required): when `antecedent_raw is None` (an SRM memory
    antecedent has no raw text), the implementation MUST fall back to scoring
    from `antecedent_repr` directly rather than joint text encoding. Design the
    cross-encoder with this dual input from the start.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("wire cross-encoder tomorrow on A100")

    def __call__(
        self,
        anaphor_repr: Tensor,
        antecedent_repr: Tensor,
        dist_feat: Tensor,
        *,
        anaphor_raw: Optional[Any] = None,
        antecedent_raw: Optional[Any] = None,
    ) -> Tensor:
        raise NotImplementedError("wire cross-encoder tomorrow on A100")
