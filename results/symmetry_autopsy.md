# Solar Ring Memory — Scorer Symmetry Autopsy

**Honest falsification record.** The gravitational scorer `Phi = lambda*G*C*R*(1-BH)` is permanently abandoned; this document proves *why* it fails in-window and is **not** an attempt to revive it. Where the data contradicted the pre-registered expectation, the data is reported as-is.

## Thesis

In-window pronoun resolution requires the model to *decide between competing candidates*. Once the hardcoded `SUBJ_SET` name-list is removed, the Solar Spring gravitational field is **exactly invariant** to which candidate is under test, so it expresses no preference and collapses to chance. This is the mirror image of cross-window entity **merging**, which works precisely because it makes *no* candidate decision — it only fuses references.

## Method

- Checkpoint: `checkpoints/winograd80_best.pt` (spring + head, `strict=False`, CPU)
- Scoring entry point: `WinogradSpringModel.score_from_vecs` (real eval path, backward attention candidate->pronoun)
- Cheat disabled: `build_concepts()` overridden so `SUBJ_SET` is empty — every token gets neutral `pos_idx=3`; no name / surface-form matching remains
- WSC273 (`WillHeld/wsc273`, test, 273 ex) treated as SEALED: run once, nothing tuned against it

## Key correction to the naive story

The pre-registered expectation was *mean score gap ~ 0*. That is **FALSIFIED**: the scorer emits large, very different numbers for the two candidates. The symmetry is not in the scores; it is in the **physics**. MiniLM embeddings are unit-norm, so every semantic mass, every force term (`lambda,G,C,R,BH`) and the entire attention matrix `A` are bit-for-bit identical between the two candidate sentences. The identical attention weight `A[pronoun,candidate]` then pulls the candidate's *raw lexical embedding* into the pronoun output — a word lookup, not a coreference decision — producing large gaps that are **uncorrelated with correctness**. Net result over the sealed set: chance.

## Results

### Symmetry test (first 50 sealed schemas)

| metric | value | pre-registered | verdict |
|---|---|---|---|
| mean \|score_correct - score_wrong\| | 7.1052 | ~0 | **falsified (gap is large)** |
| fraction with \|gap\| < 0.001 | 0.0% | high | falsified |
| accuracy (first 50) | 62.0% (95% CI 48.2–74.1%) | ~50% | CI includes 50% (subsample noise) |
| **max \|A_correct - A_wrong\|** (field) | **3.0e-07** | ~0 | **confirmed: field is candidate-invariant** |

The *field-space* metric is the real proof of symmetry: the gravitational attention is invariant to the candidate.

### Where symmetry lives / breaks (trace, 5 schemas)

Max difference between the correct- and wrong-candidate runs for each physics term (all ~0 = candidate-invariant):

| physics term | max\|correct - wrong\| |
|---|---|
| conf_weight | 0.00e+00 |
| F_micro | 2.98e-08 |
| F_macro | 2.98e-08 |
| F_spring | 0.00e+00 |
| F_orbital | 2.38e-07 |
| F_ns | 0.00e+00 |
| F_lagrange | 0.00e+00 |

Every physics term is invariant. `out[candidate]` is also invariant (the attention diagonal is zeroed, so a candidate's own value never reaches its own slot). The **only** divergence is `out[pronoun]`, where the candidate-invariant weight `A[pronoun,candidate]` multiplies the raw MiniLM embedding of whichever word was appended. That lexical residual — carrying no relational signal — is the sole input that differs, and it is uncorrelated with the correct antecedent.

## Finding 2 — the candidate is never in the representation

A **second, independent** mechanical cause, separate from the field symmetry above. Even setting the symmetric field aside, the candidate's own value never enters its own representation, so swapping candidates barely changes `out[candidate]`.

**Cause (cited):** `solar_ring/solar_spring.py:280` — `scores.fill_diagonal_(0)` zeroes the self-score before the softmax, so `A[c,c] ~ 0` and `out[candidate] = W_out(sum_j A[c,j] V[j])` excludes the candidate's own value `V[c]`. The appended candidate token contributes to nothing but itself, and it is excluded from itself.

**Demonstration (5 sealed schemas):** L2 change in `out[candidate]` vs `out[pronoun]` under a correct/wrong candidate swap.

| schema (corr/wrong) | A[c,c] | \|Δout[cand]\| | \|Δout[pron]\| | \|out[cand]\| |
|---|---|---|---|---|
| city/demonstrators | 6.76e-04 | 8.346e-03 | 9.099e+00 | 7.091 |
| demonstrators/city | 6.76e-04 | 8.493e-03 | 9.259e+00 | 7.313 |
| trophy/suitcase | 6.74e-04 | 4.254e-03 | 3.848e+00 | 1.586 |
| suitcase/trophy | 6.74e-04 | 4.153e-03 | 3.757e+00 | 1.933 |
| joan/susan | 6.74e-04 | 4.846e-03 | 5.275e+00 | 2.517 |
| **mean** | | **6.018e-03** | **6.248e+00** | |

`out[candidate]` moves by ~6.0e-03 under the swap (≈ 0 against representation norms of 1.6–7.3), while `out[pronoun]` moves ~1038× more. The candidate is, for scoring purposes, absent from its own slot.

> **SRM couldn't decide for two independent reasons — the scoring field is candidate-invariant AND the candidate is never in the representation.**

### Sealed WSC273 full run (the arbiter)

- Accuracy (all 273): **138/273 = 50.5%**
- Delta vs 50% chance: **+0.5 pp**
- Reproduces the recorded ~49.8% null.

> Guardrail OK: the sealed full-set arbiter is at chance. (The first-50 subsample read 62.0%, but its 95% CI includes 50% and the field is provably candidate-invariant, so it is sampling noise, not a leaked cheat.)

## Conclusion

The Solar Ring scorer fails **in-window** because it must choose between competing candidates while its gravitational field is invariant under swapping them: identical unit-norm masses + identical layout => identical field => identical attention => no preference => chance accuracy. The scorer's large per-item gaps come only from reading whichever raw word-embedding was appended, which carries no coreference signal. The one thing that ever broke the tie was the `SUBJ_SET` name-list — surface-form identity, not physics. Cross-window **merging** avoids this failure mode entirely because it never has to pick a winner.
