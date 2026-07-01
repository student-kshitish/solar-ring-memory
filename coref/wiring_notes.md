# Wiring seam — SRM memory → existing relational decider

**Status: design note only. No training, no full implementation.** This
describes the seam where Solar Ring Memory (SRM) hands cross-window entity
representations to the **already-existing** asymmetric antecedent scorer at
`coref/model.py:pairwise_scores` (the `cat([gi, gj, gi*gj, dist_feat])` head,
line ~130). Implementation + fitting is tomorrow's A100 work.

## Division of labour (fixed)

- **SRM = merging + memory ONLY.** It carries compressed entity state across
  window boundaries. It does **not** decide antecedents.
- **Deciding lives in the trained pair scorer** (`pairwise_scores` today; an
  A100 cross-encoder tomorrow). The gravitational field is provably symmetric
  (see `results/symmetry_autopsy.md`, Findings 1 & 2) — no physics-only fix is
  attempted or wanted.

## What SRM hands over

| Source | Symbol | Shape | Notes |
|---|---|---|---|
| `solar_ring/sun_state.py:23` (`SunState.state`) | per-paragraph "sun" | `[300]` | `D_MODEL=300` (GloVe-300d), updated by `SunState.fuse()`; one per processed paragraph |
| `solar_ring/multi_solar_system.py:90` (`gravity_waves[i]`) | compressed hand-off | `[300]` | `clone()` of a finalized `SunState.state`; the cross-system carrier |
| `solar_ring/unified_memory.py` (`entities[name]['vec']`) | per-entity vector | `[300]` | persistent entity store, `d=300`; preferred granularity for coref (one repr per entity, not per paragraph) |

The natural unit for coreference is **one representation per cross-window
entity**, so `UnifiedMemory` entity vectors are the primary feed; Sun-State /
gravity-waves are the coarser paragraph-level fallback.

Handoff record (proposed dataclass, not yet built):

```python
@dataclass
class SrmMention:
    vec: Tensor            # [300]  SRM entity/sun representation
    entity_id: str         # stable id for cluster bookkeeping
    confidence: float      # SRM merge confidence -> seeds mention_score
    doc_pos: int           # global word-start of the entity's LAST mention,
                           # used only for the distance bucket + causal order
```

## Where it enters `pairwise_scores`

Current signature (unchanged, `coref/model.py`):

```python
def pairwise_scores(
    g_pruned: Tensor,             # [M, P]   P = span_proj_dim = 256
    mention_scores_pruned: Tensor,# [M]
    starts_pruned: Tensor,        # [M]      word-start idx, document order
) -> Tensor:                      # [M, M+1] col j<i = antecedent j; col M = dummy
    gi = g_pruned.unsqueeze(1)    # ROW  i = anaphor  (current mention)
    gj = g_pruned.unsqueeze(0)    # COL  j = antecedent (earlier mention)
    # causal mask keeps j < i  ->  antecedents strictly precede anaphors
```

- Anaphor `gi` = a mention **in the current chunk** (RoBERTa span repr).
- Antecedent `gj` = an earlier mention. A cross-window entity lives in a
  **previous** chunk, so **the SRM representation enters on the `gj`
  (antecedent) side.**

### Seam: SRM entities as virtual antecedents

SRM vectors are `[300]`; the pair space is `P=256`. Project, then **prepend**
the SRM antecedents (they precede all current-chunk mentions in document order,
so the existing causal mask lets current mentions attend to them with no mask
change):

```python
class SrmAntecedentProjector(nn.Module):
    """300-d SRM entity vec -> 256-d coref pair space. Trained tomorrow."""
    def __init__(self, srm_dim: int = 300, pair_dim: int = 256):
        self.proj = nn.Linear(srm_dim, pair_dim)
    def forward(self, srm_vecs: Tensor) -> Tensor:   # [N,300] -> [N,256]
        ...

def inject_cross_chunk_antecedents(
    g_pruned: Tensor,              # [M,256] current-chunk mentions
    mention_scores_pruned: Tensor, # [M]
    starts_pruned: Tensor,         # [M]
    srm_mentions: list[SrmMention],
    projector: SrmAntecedentProjector,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepend N projected SRM entities as virtual antecedents.
    Returns (g_ext [N+M,256], ms_ext [N+M], starts_ext [N+M]) with the SRM
    block first and starts_ext assigning them earlier positions than any
    current-chunk start (so the causal tril keeps them strictly antecedent).
    mention_score for an SRM row is seeded from SrmMention.confidence.
    Then call the UNCHANGED model.pairwise_scores(g_ext, ms_ext, starts_ext)."""
    ...
```

Minimal-glue contract for this seam:
- `g_ext` block order = `[SRM antecedents ...][current-chunk mentions ...]`.
- `starts_ext` for SRM rows < min(current-chunk starts) → causal mask already
  correct, **no edit to `pairwise_scores`.**
- Distance bucket: cross-chunk gaps are large; either reuse `distance_bucket`
  (it saturates at the top edge) or add one dedicated "cross-window" bucket to
  `DIST_BUCKET_EDGES` (embedding table grows by one row — a config change, not
  logic).
- SRM rows are **antecedent-only**: they should never appear as anaphor `gi`
  (they carry no current-chunk span). Enforce by slicing predictions to the
  current-chunk rows, or by masking the SRM rows out of the anaphor axis.

## Drop-in contract for tomorrow's A100 pair scorer

Keep the boundary at the **pair-scoring function**, so the FFN can be swapped
for a cross-encoder with no change to the injection glue:

```python
class PairScorer(Protocol):
    def __call__(
        self,
        anaphor_repr: Tensor,      # [..., P]   gi
        antecedent_repr: Tensor,   # [..., P]   gj  (may be an SRM entity)
        dist_feat: Tensor,         # [..., D_dist]
        *,
        anaphor_raw: SpanRef | None = None,      # raw text span, if available
        antecedent_raw: SpanRef | None = None,   # None for SRM memory antecedents
    ) -> Tensor:                   # [...] scalar score per pair
        ...
```

- **Today's implementation** = `pair_ffn(cat([gi, gj, gi*gj, dist_feat]))`;
  input dim `3*256 + 20 = 788`. Ignores the `*_raw` kwargs.
- **Tomorrow's cross-encoder** consumes `*_raw` when present (joint token
  encoding of the two spans). **Hard constraint:** SRM antecedents are
  compressed vectors with **no raw text**, so the cross-encoder **must** accept
  a vector-only antecedent path (`antecedent_raw=None` ⇒ fall back to
  `antecedent_repr`). Design the cross-encoder with this dual input from the
  start; otherwise cross-window links can't be scored.

## Guardrails carried into tomorrow

- No physics-only "fix." The field is symmetric (Findings 1 & 2); deciding is
  the pair scorer's job, full stop.
- SRM stays merging + memory. Its vectors are inputs to the decider, never the
  decider.
- Any evaluation stays **sealed-once**: do not tune on WSC273 or the sealed
  books. Fit the projector + pair scorer on training coref data only.

## A100 checklist (tomorrow)

> Steps 1–2 implemented (CPU, untrained plumbing); 3–6 are A100 work; path is
> unproven until step 4 validates space alignment.

- **Step 3 (A100) — train the projector + pair head jointly.** Fit
  `SrmAntecedentProjector` (300→256) together with `pair_ffn` on coref data
  containing cross-window links. A randomly-initialised projector maps SRM
  vectors to noise, so the CPU plumbing (`gather_srm_antecedents`,
  `build_virtual_antecedents`, `prepend_to_pairwise`) produces correct shapes
  but **meaningless values until this training runs**.

- **Step 4 (A100) — RISK: validate that the 300-d SRM space aligns with the
  256-d span space.** SRM vectors (GloVe/physics-derived) and RoBERTa span
  reprs are unrelated spaces; a single `Linear` may be too weak to bridge them.
  **Measure cross-window dev F1 before believing anything.** If it is poor,
  upgrade the projector to an MLP or add contrastive SRM↔span alignment. This is
  a **live falsification risk** — the whole cross-window path could be a clean
  null, and that is an acceptable, honest outcome to report.

- **Step 5 (config) — distance bucket for huge cross-window gaps.** Confirm
  `distance_bucket` saturates gracefully for the large (virtual) gaps between an
  SRM antecedent and a current-chunk anaphor, or add one dedicated
  "cross-window" bucket to `DIST_BUCKET_EDGES` (grows the embedding table by one
  row — a config change, not logic).

- **Step 6 (A100) — sealed eval.** Evaluate on BookCoref documents where the
  antecedent lives in an earlier chunk, with SRM memory populated. Fit on the
  train split only; keep the eval **sealed-once**. WSC273 and the sealed books
  stay sealed — no tuning against them.
