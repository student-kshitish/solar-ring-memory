# Research direction — object-centric stateful stream computation

**Design note. A direction, not a results claim. Nothing here is a victory
statement; the headline is falsification-proof by design (a measured curve, not
a win).**

## 1. The reframe

A transformer is **token-centric and stateless**: given a window of tokens it
recomputes an output, commits nothing, and represents entities only as
distributed activations that vanish the moment the window slides. SRM — minus
the permanently retired gravitational scorer — is **object-centric and
stateful**: it maintains a discrete, inspectable, editable *set of entities* and
their relations, updated incrementally as text streams past, at fixed per-token
cost. Everything defensible about this project flows from **object-centric
statefulness**; **nothing** defensible flows from the physics. The scorer is
falsified (`results/symmetry_autopsy.md`); the memory is the contribution.

## 2. What transformers structurally lack (with the competition kept in view)

| Gap in vanilla transformers | Who already attacks it | Honest standing |
|---|---|---|
| Fixed context window / quadratic attention | Mamba/SSMs, RWKV, Ring Attention, long-context models, RAG | **Crowded — NOT a novel niche alone.** |
| No persistent state *between* inferences | Memory networks, DNC, retrieval/vector stores | A **merge-based, compressed stream memory** is a specific, under-explored point. |
| No discrete, addressable entity to read / merge / audit / correct | Sparse: EntNet, neuro-symbolic, KG-augmented LMs | **Strongest, cleanest differentiator.** |
| No cheap online fact update without retraining | Model-editing / knowledge-editing, unlearning | Ties directly into an active field. |
| No explicit contradiction tracking | Mostly absent in LMs | `UnifiedMemory` already tracks contradictions; transformers hallucinate silently. |

**Do NOT stake the claim on "unlimited context beats GPT." Stake it on discrete,
mergeable, auditable entity memory.**

## 3. Highest-value direction — the compression–fidelity frontier

The core question a transformer cannot even pose about itself:

> **How much can per-chunk state be compressed before entity identity degrades,
> and how does that curve compare to full-attention cost?**

Fixed-memory streaming *has* this knob; full attention does not. Measuring the
curve is a contribution independent of who wins.

**Experiment design.**
- **x-axis:** bytes-of-memory-per-token (sweep the compression level of the
  per-chunk Sun-State / entity representation).
- **y-axis:** coreference-chain fidelity (CoNLL F1 / B³) — does entity identity
  survive across window boundaries?
- **Testbed:** BookCoref (book-length documents, chains that exceed any window).
- **Baseline overlay:** full-attention cost/accuracy points on the same axes.

**Stated plainly: the curve is a contribution even if SRM loses at every point.**
It is a *measurement* of the identity-vs-memory tradeoff, not a victory claim.
Deciding (which mention links to which) is delegated to the trained pair scorer
(`coref/wiring_notes.md`); **memory's job is permanence, not choice.**

## 4. Ranked extra directions (defensibility, not ambition)

1. **Compression–fidelity frontier** — do this first; measured curve, self-justifying.
2. **Editable / auditable memory** — knowledge editing, provenance, targeted
   unlearning, contradiction repair; enabled because entities are discrete and
   addressable (transformers can't do this cheaply).
3. **Hybrid: memory feeds a transformer/decider** — the `wire_srm_into_coref`
   seam; **lowest-risk win**, framed as a fixed-cost memory layer and measured
   as a *delta* on long-document coreference.
4. **Falsification-as-contribution** — `symmetry_autopsy` written up as a
   cautionary/analysis paper: why symmetric energy/gravitational scorers cannot
   make coreference decisions. Rigorous negative results are rare and useful.
5. **Theory: asymmetry as a necessary condition for decision** — formalize the
   law (decision layers require functions non-invariant under candidate
   permutation) and connect it to permutation-invariance / equivariance theory
   (Deep Sets, GNN symmetry).

## 5. Honest risk (required)

- **The cross-window path may be a CLEAN NULL.** If a fixed-size merge memory
  cannot hold entity identity at book scale (the step-4 alignment risk in
  `wiring_notes.md`: 300-d SRM space may not align with the span space), that
  negative result **is the finding** — measured, honest, and publishable. The
  compression–fidelity framing survives a null because a flat or collapsing
  curve is still a curve.
- **Do not revive the scorer.** Do not frame any of this as "beats GPT" or
  "context supremacy" — that invites the exact refutation performed tonight.
  SRM = memory + merging only; deciding lives in the trained relational scorer.

## 6. Thesis sentence

> We study language processing as **object-centric, stateful stream
> computation** — maintaining a **discrete, mergeable, inspectable entity
> memory** across book-length text at **fixed per-token cost** — and
> **characterize the compression–fidelity frontier of identity maintenance**, a
> regime transformers structurally cannot represent. **Deciding is delegated to
> a trained relational scorer; the memory's job is permanence, not choice.**
