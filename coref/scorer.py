"""From-scratch CoNLL coreference scoring: MUC, B3, CEAFe (phi4).

Clusters are represented as lists of mentions, where a mention is any
hashable id (here: (sent_idx, start, end) tuples). All three metrics
are computed per-document as (numerator, denominator) pairs for both
precision and recall, so that corpus-level scores are obtained by
summing numerators/denominators across documents before dividing
(standard "micro" aggregation used by the official CoNLL scorer), and
bootstrap CIs can resample documents and re-aggregate cheaply.
"""
from __future__ import annotations

from typing import Hashable, Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment

Cluster = list[Hashable]


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r)


# ---------------------------------------------------------------- MUC
def muc_counts(gold: Sequence[Cluster], resp: Sequence[Cluster]):
    """Return (recall_num, recall_den, prec_num, prec_den)."""

    def links_lost(key_clusters, response_clusters):
        # number of response clusters intersecting each key cluster
        mention_to_resp = {}
        for ci, c in enumerate(response_clusters):
            for m in c:
                mention_to_resp[m] = ci
        num = 0.0
        den = 0.0
        for c in key_clusters:
            if len(c) < 2:
                continue
            den += len(c) - 1
            partitions = {mention_to_resp.get(m, ("_unmatched", m)) for m in c}
            num += len(c) - len(partitions)
        return num, den

    r_num, r_den = links_lost(gold, resp)
    p_num, p_den = links_lost(resp, gold)
    return r_num, r_den, p_num, p_den


# ----------------------------------------------------------------- B3
def b3_counts(gold: Sequence[Cluster], resp: Sequence[Cluster]):
    mention_to_gold: dict[Hashable, frozenset] = {}
    for c in gold:
        fc = frozenset(c)
        for m in c:
            mention_to_gold[m] = fc
    mention_to_resp: dict[Hashable, frozenset] = {}
    for c in resp:
        fc = frozenset(c)
        for m in c:
            mention_to_resp[m] = fc

    r_num = 0.0
    r_den = 0.0
    for c in gold:
        for m in c:
            rc = mention_to_resp.get(m, frozenset())
            r_num += len(rc & frozenset(c)) / len(c)
            r_den += 1

    p_num = 0.0
    p_den = 0.0
    for c in resp:
        for m in c:
            gc = mention_to_gold.get(m, frozenset())
            p_num += len(gc & frozenset(c)) / len(c)
            p_den += 1

    return r_num, r_den, p_num, p_den


# -------------------------------------------------------------- CEAFe
def ceafe_counts(gold: Sequence[Cluster], resp: Sequence[Cluster]):
    if len(gold) == 0 or len(resp) == 0:
        return 0.0, len(gold), 0.0, len(resp)
    sets_g = [frozenset(c) for c in gold]
    sets_r = [frozenset(c) for c in resp]
    sim = np.zeros((len(sets_g), len(sets_r)))
    for i, g in enumerate(sets_g):
        for j, r in enumerate(sets_r):
            denom = len(g) + len(r)
            sim[i, j] = (2 * len(g & r) / denom) if denom > 0 else 0.0
    row_ind, col_ind = linear_sum_assignment(-sim)
    matched = sim[row_ind, col_ind].sum()
    return matched, len(sets_g), matched, len(sets_r)


METRICS = {"muc": muc_counts, "b3": b3_counts, "ceafe": ceafe_counts}


def score_document(gold: Sequence[Cluster], resp: Sequence[Cluster]) -> dict:
    """Per-document (r_num, r_den, p_num, p_den) for each metric."""
    out = {}
    for name, fn in METRICS.items():
        out[name] = fn(gold, resp)
    return out


def aggregate(doc_scores: Sequence[dict]) -> dict:
    """Sum per-doc counts then compute P/R/F1 per metric + CoNLL average F1."""
    out = {}
    for name in METRICS:
        r_num = sum(d[name][0] for d in doc_scores)
        r_den = sum(d[name][1] for d in doc_scores)
        p_num = sum(d[name][2] for d in doc_scores)
        p_den = sum(d[name][3] for d in doc_scores)
        r = _safe_div(r_num, r_den)
        p = _safe_div(p_num, p_den)
        out[name] = {"precision": p, "recall": r, "f1": f1(p, r)}
    out["conll_f1"] = sum(out[m]["f1"] for m in METRICS) / len(METRICS)
    return out


def bootstrap_ci(doc_scores: Sequence[dict], n_boot: int = 1000, seed: int = 0):
    """Bootstrap resample documents (with replacement) to get a 95% CI on
    each metric's F1 and on the averaged CoNLL F1."""
    rng = np.random.default_rng(seed)
    n = len(doc_scores)
    samples = {name: [] for name in METRICS}
    conll = []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        boot = [doc_scores[i] for i in idx]
        agg = aggregate(boot)
        for name in METRICS:
            samples[name].append(agg[name]["f1"])
        conll.append(agg["conll_f1"])
    ci = {}
    for name in METRICS:
        arr = np.array(samples[name])
        ci[name] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    conll_arr = np.array(conll)
    ci["conll_f1"] = (float(np.percentile(conll_arr, 2.5)), float(np.percentile(conll_arr, 97.5)))
    return ci
