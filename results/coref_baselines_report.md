# PreCo Coreference Baselines — Honest Pre-Memory-Layer Assessment

**Goal:** measure how much of PreCo coreference resolution plain fine-tuned
RoBERTa-base already solves with three different amounts of encoder
context, *before* building any Solar Ring memory layer, so any later claim
that the memory layer "adds accuracy" has a real, sealed baseline to beat.

No memory layer was built. Nothing was pushed. The 500-doc PreCo
`validation` split was sealed to `data/test_holdout_preco.json` before any
training and evaluated **exactly once**, at the very end, for the numbers
below.

## Setup

- **Model**: one shared architecture (`coref/model.py`) for all three
  baselines — a span-based mention detector (handles nested/overlapping
  spans, unlike BIO tagging) + pairwise antecedent linker, in the style of
  Lee et al. 2017 end-to-end coreference, fine-tuned end-to-end on
  `roberta-base`. Marginal log-likelihood antecedent loss + BCE mention
  loss, top-40%-by-score span pruning, max span width 10 words.
- **The only thing that differs between baselines is the context chunk**
  the encoder sees and mentions are allowed to link within — every token
  is encoded exactly once per document in all three, so training cost is
  comparable:
  - **(a) sentence**: one sentence per chunk. No cross-sentence link is
    architecturally possible.
  - **(b) window5**: non-overlapping 5-sentence chunks.
  - **(c) fulldoc**: chunks packed greedily up to 512 subword tokens
    (≈ the whole document for 96% of PreCo docs, which average 334
    tokens/25 sentences).
- **Splits**: train on `train[:32958]`, model-selected on a fixed
  300-doc subsample of the dev split (`train[32958:36620]`, 3,662 docs)
  via early stopping (patience 5) and a wall-clock budget per baseline
  (~80–130 min on a single RTX 5050 8GB). Final numbers below are one
  evaluation of the selected checkpoint on the sealed 500.
- **Compute note (honest disclosure)**: each baseline trained for
  ~1–1.7 epochs before plateauing/hitting its time budget, not full
  convergence. The `sentence` baseline crashed once on CUDA OOM
  (allocator fragmentation from highly variable sentence lengths) and was
  restarted with a fix (`expandable_segments`, smaller batch, OOM-safe
  skip-batch handling) — final reported number is from the clean rerun.
- **CoNLL scorer**: MUC / B³ / CEAFe (φ4 + Hungarian matching) reimplemented
  from scratch (`coref/scorer.py`), validated against the canonical Vilain
  et al. 1995 MUC worked example and a perfect-match sanity check.
- **Cross-sentence bucket metric**: pairwise link P/R/F1 restricted to
  pairs of *gold* mention spans, computed only over pairs whose sentence
  distance falls in the bucket — i.e., "given two real mentions this far
  apart, did the system put them in the same predicted cluster?"

## Results on the sealed 500-doc holdout (n=500, evaluated once)

| Baseline | CoNLL F1 (95% CI) | MUC F1 | B³ F1 | CEAFe F1 |
|---|---|---|---|---|
| (a) sentence | 0.6133 (0.6069–0.6189) | 0.483 | 0.698 | 0.659 |
| (b) window5 | 0.7313 (0.7266–0.7356) | 0.740 | 0.739 | 0.716 |
| (c) fulldoc | **0.7997** (0.7951–0.8043) | 0.858 | 0.796 | 0.746 |

Confidence intervals don't overlap between any pair of baselines — more
context is unambiguously better, and **(c) full-document is the strongest
baseline, and the real bar a memory layer has to beat.**

### Cross-sentence-only pairwise link F1, by gold mention distance

| Baseline | 1–3 sents | 4–9 sents | 10+ sents |
|---|---|---|---|
| (a) sentence | 0.000 (n=31,990 pairs) | 0.000 (n=38,879) | 0.000 (n=53,347) |
| (b) window5 | 0.716 | 0.084 | 0.000 |
| (c) fulldoc | 0.901 | 0.891 | **0.800** (P=0.916, R=0.711) |

This is exactly the expected structural signature: (a) cannot link across
sentences at all by construction (0 everywhere it should be 0); (b) can
only recover links inside its 4-sentence look-back, so it falls off a
cliff right where the 4–9 bucket starts and is architecturally incapable
of the 10+ bucket; (c) sees the (almost always complete) document and
keeps working, with precision staying high (0.92) even at 10+ sentences
while recall gradually erodes (0.71).

## Step 3 — honest assessment

**Strongest baseline:** (c) full-document, by a wide and statistically
significant margin (0.80 vs 0.73 vs 0.61 CoNLL F1, non-overlapping CIs).

**How much does full-document RoBERTa already get right at 10+ sentences?**
80.0% F1 (91.6% precision / 71.1% recall) on cross-sentence links at
10+ sentences of gold distance — using nothing but plain bidirectional
self-attention over up to 512 tokens, no specialized memory mechanism at
all. That's the overwhelming majority of long-range coreference signal in
this dataset, recovered "for free" by an off-the-shelf transformer encoder
fine-tuned with a standard antecedent-ranking head.

**Is there room for a memory layer to add accuracy, or is its honest
contribution interpretability/efficiency only?**

On PreCo specifically: **mostly interpretability + memory-efficiency, not
accuracy.** Two things drive that conclusion:

1. **96% of PreCo documents fit entirely inside a 512-token window** (avg
   334 tokens / 25 sentences). For those documents "full-document" *is*
   the actual full document — there's no information a memory layer could
   surface that the encoder doesn't already have direct attention access
   to. The case for a memory layer's accuracy benefit only has real teeth
   on documents that *exceed* the encoder's context window, forcing
   truncation/chunking — exactly the regime where (b) window5 already
   shows the catastrophic failure mode (0% F1 at 10+ sentences, because
   the two mentions simply land in different chunks that never talk to
   each other). PreCo barely tests that regime.
2. **The remaining gap at 10+ sentences is a recall gap inside a single
   attention window where all tokens are already visible** (precision
   stays at 0.92; recall drops from ~0.88 at short range to 0.71 at long
   range). That's the signature of an under-trained or representation
   problem (the model has the information but isn't using it as
   confidently at distance), not a memory-capacity/context-overflow
   problem. A structured memory layer doesn't have a natural mechanism to
   fix "the transformer already saw this token but discounted it" — more
   training steps, better long-distance features, or a stronger
   antecedent-ranking head are the more direct fixes for *that* gap.

**Bottom line:** if a memory layer is built, the experiment that would
actually test its value is documents *longer than 512 tokens* (forcing
today's "full-document" baseline to truncate/chunk and lose long-range
links the way window5 already does at shorter range) — not PreCo as
distributed, where the plain fine-tuned transformer already gets within
~20 points of ceiling on the hardest distance bucket just by looking at
the page. On PreCo as-is, the defensible pitch for a memory layer is
interpretability (explicit, inspectable slots vs. opaque attention) and
memory footprint, not raw coreference accuracy.

Raw numbers: `results/coref_baselines_report.json`. Sealed test set:
`data/test_holdout_preco.json` (never read until this evaluation).
