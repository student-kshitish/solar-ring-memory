"""
BookCoref chunked baseline — lengths-only approach.

Uses the gold annotation (mention token positions) + the sentence lengths file
to compute chunk boundaries and all metrics WITHOUT reconstructing full text.
This is valid because:
  - Mention positions are flat token indices in the final annotated doc
  - Sentence lengths tell us exactly where each sentence ends
  - Chunk boundaries follow sentence boundaries at CHUNK_SIZE tokens

STEP 0: Seals gold test books to results/bookcoref_gold_sealed.json.
STEP 1: Computes chunk boundaries from sentence lengths at CHUNK_SIZE=512.
STEP 2: For each mention pair in gold clusters, checks within vs cross-chunk.
        Runs oracle-mention chunked baseline (no cross-chunk linking).
        Scores with CoNLL F1 (MUC + B³ + CEAF-e).
STEP 3: Reports the gap: baseline → ceiling (perfect cross-chunk = gold).

No memory layer. No cross-chunk linking. No text download needed.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEALED_PATH = RESULTS_DIR / "bookcoref_gold_sealed.json"
BASELINE_PATH = RESULTS_DIR / "bookcoref_baseline_results.json"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "datasets" / "downloads"

TEST_JSONL   = CACHE_DIR / "f7f0d411d70d2531eb0935fa3cd69683125c645b94d490183126497304497d97"
LENGTHS_FILE = CACHE_DIR / "fcb01fa02443a80b9f6a9b0fa0cc56de9af0b21d8b4a5297ea00cfd2ff30a433"

CHUNK_SIZE = 512


# ── CoNLL-F1 metrics (self-contained) ─────────────────────────────────────────

def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def muc(pred, gold):
    def _score(key, response):
        resp_sets = response
        total = sum(len(c) for c in key)
        if total == 0:
            return 0.0
        num = 0
        for kc in key:
            seen_resp = set()
            unmatched = 0
            for m in kc:
                found = False
                for i, rc in enumerate(resp_sets):
                    if m in rc:
                        seen_resp.add(i)
                        found = True
                        break
                if not found:
                    unmatched += 1
            partitions = len(seen_resp) + unmatched
            num += len(kc) - max(partitions, 1)
        denom = sum(len(c) - 1 for c in key)
        return num / denom if denom > 0 else 1.0
    r = _score(gold, pred)
    p = _score(pred, gold)
    return p, r, _f1(p, r)


def b_cubed(pred, gold):
    gold_map = {m: i for i, cl in enumerate(gold) for m in cl}
    pred_map = {m: i for i, cl in enumerate(pred) for m in cl}
    all_m = set(gold_map) | set(pred_map)
    if not all_m:
        return 1.0, 1.0, 1.0
    pv, rv = [], []
    for m in all_m:
        gi = gold_map.get(m)
        pi = pred_map.get(m)
        gc = gold[gi] if gi is not None else set()
        pc = pred[pi] if pi is not None else set()
        ov = gc & pc
        pv.append(len(ov) / len(pc) if pc else 0.0)
        rv.append(len(ov) / len(gc) if gc else 0.0)
    p, r = sum(pv) / len(pv), sum(rv) / len(rv)
    return p, r, _f1(p, r)


def ceafe(pred, gold):
    if not gold or not pred:
        return 0.0, 0.0, 0.0
    def phi4(g, p):
        return 2 * len(g & p) / (len(g) + len(p)) if (len(g) + len(p)) > 0 else 0.0
    cost = np.array([[phi4(g, p) for p in pred] for g in gold])
    ri, ci = linear_sum_assignment(-cost)
    score = float(cost[ri, ci].sum())
    return score / len(pred), score / len(gold), _f1(score / len(pred), score / len(gold))


def conll_f1(pred_cls, gold_cls):
    g = [set(map(tuple, c)) for c in gold_cls if c]
    p = [set(map(tuple, c)) for c in pred_cls if c]
    mp, mr, mf = muc(p, g)
    bp, br, bf = b_cubed(p, g)
    ep, er, ef = ceafe(p, g)
    avg = (mf + bf + ef) / 3
    return {
        "muc":   {"p": round(mp*100,2), "r": round(mr*100,2), "f": round(mf*100,2)},
        "b3":    {"p": round(bp*100,2), "r": round(br*100,2), "f": round(bf*100,2)},
        "ceafe": {"p": round(ep*100,2), "r": round(er*100,2), "f": round(ef*100,2)},
        "conll_f1": round(avg * 100, 2),
    }


# ── chunking from sentence lengths (no text needed) ───────────────────────────

def chunk_starts_from_lengths(sent_lengths: list[int], chunk_size: int) -> list[int]:
    """
    Compute chunk start token indices (sentence-boundary-aligned).
    Each chunk is as close to chunk_size tokens as possible without
    splitting a sentence.
    """
    starts = [0]
    pos = 0
    total = sum(sent_lengths)
    for slen in sent_lengths:
        pos += slen
        if pos - starts[-1] >= chunk_size and pos < total:
            starts.append(pos)
    return starts


def mention_chunk(tok_start: int, starts: list[int]) -> int:
    """Return which chunk a mention (by start token) belongs to."""
    chunk = 0
    for i, s in enumerate(starts):
        if tok_start >= s:
            chunk = i
    return chunk


# ── analysis ───────────────────────────────────────────────────────────────────

def cross_chunk_stats(gold_cls, starts, total_tokens):
    within = cross = 0
    for cl in gold_cls:
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                ci = mention_chunk(cl[i][0], starts)
                cj = mention_chunk(cl[j][0], starts)
                if ci == cj:
                    within += 1
                else:
                    cross += 1
    total = within + cross
    return {
        "within_pairs": within,
        "cross_pairs": cross,
        "cross_fraction_pct": round(cross / total * 100, 2) if total else 0.0,
        "n_chunks": len(starts),
        "total_tokens": total_tokens,
    }


def chunked_baseline_predict(gold_cls, starts):
    """
    Oracle-mention chunked baseline: all gold mentions known, but only
    within-chunk mentions are linked.  Upper bound for any chunked model.
    """
    pred = defaultdict(list)
    for cl_id, cl in enumerate(gold_cls):
        for m in cl:
            cid = mention_chunk(m[0], starts)
            pred[(cid, cl_id)].append(m)
    return list(pred.values())


def clusters_from_characters(characters):
    return [[(m[0], m[1]) for m in ch["mentions"]] for ch in characters if ch["mentions"]]


# ── per-character cross-chunk analysis ────────────────────────────────────────

def per_entity_cross_chunk(gold_cls, starts):
    rows = []
    for i, cl in enumerate(gold_cls):
        chunks_seen = set(mention_chunk(m[0], starts) for m in cl)
        cross = sum(
            1
            for a in range(len(cl))
            for b in range(a + 1, len(cl))
            if mention_chunk(cl[a][0], starts) != mention_chunk(cl[b][0], starts)
        )
        within = len(cl) * (len(cl) - 1) // 2 - cross
        rows.append({
            "entity_idx": i,
            "n_mentions": len(cl),
            "n_chunks_span": len(chunks_seen),
            "within_pairs": within,
            "cross_pairs": cross,
        })
    return rows


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading sentence lengths ...")
    with open(LENGTHS_FILE) as f:
        lengths = json.load(f)

    print("Loading gold test annotations ...")
    gold_books = []
    with open(TEST_JSONL) as f:
        for line in f:
            gold_books.append(json.loads(line))
    print(f"  {len(gold_books)} gold books")

    # ── STEP 0: seal ────────────────────────────────────────────────────────
    if not SEALED_PATH.exists():
        SEALED_PATH.write_text(json.dumps(gold_books, indent=2))
        print(f"\n[STEP 0] Sealed {len(gold_books)} gold books → {SEALED_PATH}")
    else:
        print(f"\n[STEP 0] Already sealed at {SEALED_PATH}")

    all_results = {}

    for book in gold_books:
        key = book["gutenberg_key"]
        doc_key = book["doc_key"]

        if key == "0" or "animal_farm" in doc_key:
            print(f"\nSkipping {doc_key} (no public text / copyright)")
            continue

        print(f"\n{'='*64}")
        print(f"Book: {doc_key}  (Gutenberg #{key})")

        # Sentence lengths from the lengths file
        sent_lens = lengths[key]
        total_tokens = sum(sent_lens)
        n_sents = len(sent_lens)
        print(f"  {n_sents} sentences, {total_tokens:,} tokens")

        # Mention clusters from gold
        gold_cls = clusters_from_characters(book["characters"])
        n_mentions = sum(len(c) for c in gold_cls)
        print(f"  Gold: {len(gold_cls)} entity clusters, {n_mentions} mentions")

        # Validate mention indices are within range
        max_idx = max((m[1] for cl in gold_cls for m in cl), default=0)
        print(f"  Max mention token index: {max_idx:,}  (doc has {total_tokens:,} tokens)")
        if max_idx >= total_tokens:
            print(f"  WARNING: max mention index {max_idx} >= doc length {total_tokens}!")

        # ── Chunk boundaries ──────────────────────────────────────────────
        starts = chunk_starts_from_lengths(sent_lens, CHUNK_SIZE)
        chunk_ends = starts[1:] + [total_tokens]
        print(f"  Chunks: {len(starts)}")
        print(f"  Chunk sizes: min={min(e-s for s,e in zip(starts,chunk_ends))}, "
              f"max={max(e-s for s,e in zip(starts,chunk_ends))}, "
              f"mean={total_tokens//len(starts)}")

        # ── STEP 2: cross-chunk analysis ─────────────────────────────────
        stats = cross_chunk_stats(gold_cls, starts, total_tokens)
        print(f"\n  [STEP 2] Cross-chunk mention-pair analysis:")
        print(f"    within-chunk pairs : {stats['within_pairs']:,}")
        print(f"    cross-chunk  pairs : {stats['cross_pairs']:,}")
        print(f"    cross-chunk  %     : {stats['cross_fraction_pct']:.1f}%")

        # Per-entity stats (how many entities span multiple chunks?)
        entity_rows = per_entity_cross_chunk(gold_cls, starts)
        multi_chunk = sum(1 for r in entity_rows if r['n_chunks_span'] > 1)
        print(f"    entities spanning >1 chunk: {multi_chunk}/{len(gold_cls)}")
        max_span = max(r['n_chunks_span'] for r in entity_rows)
        print(f"    max chunks spanned by 1 entity: {max_span}")

        # ── STEP 2: baseline F1 ───────────────────────────────────────────
        pred_base = chunked_baseline_predict(gold_cls, starts)
        scores_base = conll_f1(pred_base, gold_cls)

        print(f"\n  [BASELINE] oracle-mention, no cross-chunk linking:")
        print(f"    MUC   P={scores_base['muc']['p']:.1f}  "
              f"R={scores_base['muc']['r']:.1f}  F={scores_base['muc']['f']:.1f}")
        print(f"    B³    P={scores_base['b3']['p']:.1f}  "
              f"R={scores_base['b3']['r']:.1f}  F={scores_base['b3']['f']:.1f}")
        print(f"    CEAFe P={scores_base['ceafe']['p']:.1f}  "
              f"R={scores_base['ceafe']['r']:.1f}  F={scores_base['ceafe']['f']:.1f}")
        print(f"    CoNLL F1 = {scores_base['conll_f1']:.1f}%")

        # ── STEP 3: ceiling ───────────────────────────────────────────────
        scores_ceil = conll_f1(gold_cls, gold_cls)
        gap = scores_ceil["conll_f1"] - scores_base["conll_f1"]
        print(f"\n  [CEILING] perfect cross-chunk linking = gold:")
        print(f"    CoNLL F1 = {scores_ceil['conll_f1']:.1f}%")
        print(f"  Memory-layer max gain: +{gap:.1f}% F1")

        all_results[doc_key] = {
            "doc_key": doc_key,
            "gutenberg_key": key,
            "total_tokens": total_tokens,
            "n_sentences": n_sents,
            "n_clusters": len(gold_cls),
            "n_mentions": n_mentions,
            "n_chunks": len(starts),
            "chunk_size": CHUNK_SIZE,
            "cross_chunk_stats": stats,
            "per_entity_stats": entity_rows,
            "baseline_scores": scores_base,
            "ceiling_scores": scores_ceil,
            "memory_layer_max_gain_f1": round(gap, 2),
        }

    # ── summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("FINAL SUMMARY")
    print(f"{'='*64}")
    for doc_key, res in all_results.items():
        cs = res["cross_chunk_stats"]
        print(f"\n{doc_key}")
        print(f"  {res['total_tokens']:,} tokens | {res['n_chunks']} chunks "
              f"| {res['n_clusters']} entities | {res['n_mentions']} mentions")
        print(f"  Cross-chunk pairs   : {cs['cross_fraction_pct']:.1f}%")
        print(f"  Baseline  CoNLL F1  : {res['baseline_scores']['conll_f1']:.1f}%  "
              f"(oracle mentions, no cross-chunk linking)")
        print(f"  Ceiling   CoNLL F1  : {res['ceiling_scores']['conll_f1']:.1f}%  "
              f"(perfect memory)")
        print(f"  Gap (max gain)      : +{res['memory_layer_max_gain_f1']:.1f}%")

    BASELINE_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {BASELINE_PATH}")

    print("\n" + "="*64)
    print("IMPORTANT NOTES")
    print("="*64)
    print("1. This is the ORACLE-MENTION baseline: all gold mention positions")
    print("   are known in advance. A real coref model (imperfect detection)")
    print("   will score LOWER. The baseline here is the upper bound for any")
    print("   chunked system.")
    print("2. The ceiling (gold=pred) is the theoretical maximum if cross-chunk")
    print("   linking is perfect. In practice a memory layer won't reach 100%,")
    print("   but any gain above baseline is a real improvement.")
    print("3. Animal Farm was excluded: no public domain Gutenberg text.")


if __name__ == "__main__":
    main()
