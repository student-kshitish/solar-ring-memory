"""Integration glue: SRM virtual antecedents -> existing pairwise scoring.

Turns SRM cross-window entities into virtual antecedent mentions and prepends
them to a chunk's pruned mention set, so the UNCHANGED
`coref/model.py:pairwise_scores` can link current-chunk anaphors to them. SRM
entities always precede current-chunk mentions in (virtual) document order, so
the existing `tril(diagonal=-1)` causal mask needs no edit.

Tonight: `build_virtual_antecedents` is real tensor plumbing (fully
implemented). `prepend_to_pairwise` is the integration point wired tomorrow.
No training, no model forward executed here.
"""
from __future__ import annotations

import torch
from torch import Tensor

from coref.srm_bridge import PAIR_DIM, SrmAntecedentProjector, SrmMention


def build_virtual_antecedents(
    srm_mentions: list[SrmMention],
    projector: SrmAntecedentProjector,
) -> tuple[Tensor, Tensor]:
    """Project SRM entities into pair space and collect their virtual starts.

    Args:
        srm_mentions: cross-window entities from `gather_srm_antecedents`.
        projector:    300 -> 256 SrmAntecedentProjector.

    Returns:
        reprs:  [M, 256] projected antecedent representations (gj side).
        starts: [M]     virtual document starts, earlier than any current-chunk
                        anaphor (carried on each SrmMention.start), so the
                        existing causal mask keeps them strictly antecedent.
    """
    if not srm_mentions:
        return (
            torch.empty(0, PAIR_DIM),
            torch.empty(0, dtype=torch.long),
        )

    stacked_300 = torch.stack([m.repr_300 for m in srm_mentions], dim=0)  # [M,300]
    reprs = projector(stacked_300)                                        # [M,256]
    starts = torch.tensor([m.start for m in srm_mentions], dtype=torch.long)  # [M]
    return reprs, starts


def prepend_to_pairwise(
    g_pruned: Tensor,
    mention_scores_pruned: Tensor,
    starts_pruned: Tensor,
    virtual_reprs: Tensor,
    virtual_starts: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepend virtual SRM antecedents to a chunk's pruned mention set.

    Returns the extended (g_ext, mention_scores_ext, starts_ext) with the SRM
    block first, ready to pass unchanged into
    `coref/model.py:pairwise_scores`. Seeding SRM mention scores (from merge
    confidence) and slicing SRM rows off the anaphor axis are wired tomorrow.
    """
    raise NotImplementedError("wire integration tomorrow on A100")
