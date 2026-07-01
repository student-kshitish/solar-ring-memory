# Reasoning architecture — conceptual design note

**Design note only. No code, no implementation, no training, nothing committed.**
Companion to `wiring_notes.md`. This is a *direction*, not a claim of a reasoning
system and not a roadmap being committed to build.

## 1. The distinction (stated once)

- **Association / merging** — "these two references are the same entity." There is
  no alternative to weigh. **SRM does this** (merging + memory).
- **Decision under competing hypotheses** — "given the context, is it *A* or *B*?"
  Choosing between alternatives from how each *relates* to the context. **This is
  reasoning, and SRM provably cannot do it** (see `results/symmetry_autopsy.md`).

## 2. The law (anchor)

> **No architecture reasons by scoring each option independently and comparing
> scalars over a context that doesn't depend on the option. Reasoning requires an
> asymmetric, relational function whose output changes when the alternatives
> change.**

Established tonight on sealed WSC273, two independent measurements:
- **Field is candidate-invariant:** `max|A_correct − A_wrong| ≈ 3e-7`.
- **Candidate is never represented:** `‖Δout[candidate]‖ ≈ 6e-3` under a
  correct/wrong swap vs `‖Δout[pronoun]‖ ≈ 6.2` — a **1038×** ratio.

Why gravity is the wrong primitive: gravitational attraction is symmetric,
`F(a,b) = F(b,a)`. Deciding is **directional** (mention → antecedent, premise →
conclusion). A symmetric primitive cannot express a directional choice, so no
amount of added physics terms can make the field reason.

## 3. The three layers

| Layer | What it adds | Primitive that unlocks it | Honest status in this repo |
|---|---|---|---|
| **L1 — one-step relational decision** | choose A vs B from their relation to context | asymmetric interaction: `q·k`, bilinear, `cat([gi, gj, gi*gj])` | **BUILT** — `coref/model.py:pairwise_scores` |
| **L2 — multi-hop / working memory** | chain decisions, carry intermediate state | read/write memory **+** a controller that re-queries it | **SUBSTRATE-ONLY** — SRM is a candidate memory substrate; the controller is **NOT BUILT** |
| **L3 — verification / self-correction** | judge whether a reasoning step is valid | a discriminator (trace → valid?), itself asymmetric | **NOT BUILT** |

The single primitive underneath all three is the one SRM lacked: an **asymmetric
relational function**. L2 and L3 are labeled truthfully — "candidate substrate"
and "not built." Nothing here is claimed to work beyond L1.

## 4. The architecture (in prose)

**SRM (merging + unbounded memory)** → hands compressed entity state to a
**relational decider (pair-scorer / cross-encoder)** → wrapped in a **controller
loop (query memory, decide, verify, repeat)**.

Reasoning emerges from the **loop over memory + relational decisions**, not from
any single component — and even frontier LLMs only *approximate* this (scale for
the decider, chain-of-thought for iteration, RL/verification for self-correction,
retrieval/tools for memory). There is no closed-form reasoning formula in any of
them. This doc names a direction; it does not assert a reasoning system exists.

## 5. The smallest real thing

The **option-1 seam is a working L1 one-step reasoner**: it decides an antecedent
from the *relation* between a mention and a candidate, with candidates read from
SRM memory — an asymmetric relational decision over an unbounded substrate.

Explicitly: the **L2 controller and L3 verifier are FUTURE, UNSCOPED** — not
tomorrow's work. **Tomorrow is coreference only** (the +9 antecedent line +
cross-encoder pair scorer), full stop.

## Guardrails

- No physics-reasoning revival. The field is provably symmetric; deciding lives
  in the trained relational scorer.
- SRM = memory + merging only. Its vectors feed the decider; they never decide.
- Every capability above is labeled **built / substrate-only / not-built**
  truthfully. This is a design note, not a build commitment.
