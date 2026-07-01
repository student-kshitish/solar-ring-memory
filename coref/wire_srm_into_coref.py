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
    virtual_scores: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepend virtual SRM antecedents to a chunk's pruned mention set.

    Pure tensor ops (cat + a boolean mask). Does NOT call `pairwise_scores` or
    any model — it only assembles the extended inputs that a downstream caller
    passes into the UNCHANGED `coref/model.py:pairwise_scores`.

    Block order is [virtual SRM antecedents][current-chunk mentions]; combined
    with their below-chunk `virtual_starts`, the existing `tril(diagonal=-1)`
    causal mask keeps SRM rows strictly antecedent with no edit.

    SRM rows are masked OFF the anaphor axis: an SRM entity is a compressed
    memory reference, never the current mention being resolved. The returned
    `anaphor_mask` (True = row may act as an anaphor) has False for every SRM
    row; the caller applies it to the anaphor (row) axis of the pair scores.

    NOTE (untrained plumbing): shapes are correct, but the SRM `virtual_reprs`
    come from a randomly-initialised projector until step 3 trains it — so the
    prepended block is meaningless in VALUE tonight. Correct shapes only.

    Args:
        g_pruned:              [M, 256] current-chunk pruned mention reprs.
        mention_scores_pruned: [M] current-chunk mention scores.
        starts_pruned:         [M] current-chunk word-start indices.
        virtual_reprs:         [V, 256] projected SRM antecedents.
        virtual_starts:        [V] below-chunk virtual starts.
        virtual_scores:        [V] SRM mention-score seeds from merge
                               confidence; None => neutral zeros.

    Returns:
        g_ext:              [V+M, 256]
        mention_scores_ext: [V+M]
        starts_ext:         [V+M]
        anaphor_mask:       [V+M] bool, False for the V SRM rows (antecedent-only)
    """
    V = virtual_reprs.size(0)

    if virtual_scores is None:
        virtual_scores = torch.zeros(V, dtype=mention_scores_pruned.dtype)

    g_ext = torch.cat([virtual_reprs.to(g_pruned), g_pruned], dim=0)
    mention_scores_ext = torch.cat(
        [virtual_scores.to(mention_scores_pruned), mention_scores_pruned], dim=0
    )
    starts_ext = torch.cat(
        [virtual_starts.to(starts_pruned), starts_pruned], dim=0
    )

    # SRM rows (first V) can only be antecedents, never the current anaphor.
    anaphor_mask = torch.ones(g_ext.size(0), dtype=torch.bool)
    anaphor_mask[:V] = False

    return g_ext, mention_scores_ext, starts_ext, anaphor_mask
