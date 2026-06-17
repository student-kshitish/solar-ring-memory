from __future__ import annotations

import torch

from coref.data import build_chunks_for_doc
from coref.engine import predict_chunk_clusters, local_span_to_global_id, global_id_for_gold
from coref.scorer import score_document, aggregate, bootstrap_ci

BUCKETS = [("1-3", 1, 3), ("4-9", 4, 9), ("10+", 10, float("inf"))]


def gold_clusters_global(mention_clusters):
    return [[global_id_for_gold(s, st, en) for (s, st, en) in cluster] for cluster in mention_clusters]


@torch.no_grad()
def predict_doc_clusters(model, ex, baseline, tokenizer, device):
    def tok_len_fn(words):
        return len(tokenizer(words, is_split_into_words=True, add_special_tokens=False)["input_ids"])

    chunks = build_chunks_for_doc(0, ex["sentences"], ex["mention_clusters"], baseline, tok_len_fn)
    pred_global = []
    for chunk in chunks:
        if not chunk.words:
            continue
        enc = tokenizer([chunk.words], is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        word_ids = enc.word_ids(0)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            (spans, mention_scores, g), = model.forward_batch(input_ids, attn, [word_ids])
        num_words = len(chunk.words)
        clusters_local = predict_chunk_clusters(model, spans, mention_scores, g, num_words)
        for cl in clusters_local:
            pred_global.append([local_span_to_global_id(chunk.word_doc_id, s, e) for (s, e) in cl])
    return pred_global


def bucket_pair_counts(gold_clusters, pred_clusters):
    mention_to_pred = {}
    for ci, c in enumerate(pred_clusters):
        for m in c:
            mention_to_pred[m] = ci
    mention_to_gold = {}
    for ci, c in enumerate(gold_clusters):
        for m in c:
            mention_to_gold[m] = ci

    counts = {name: [0, 0, 0, 0] for name, _, _ in BUCKETS}  # r_num,r_den,p_num,p_den

    def bucket_for(d):
        for name, lo, hi in BUCKETS:
            if lo <= d <= hi:
                return name
        return None

    # recall: iterate gold pairs within each gold cluster
    for c in gold_clusters:
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                mi, mj = c[i], c[j]
                d = abs(mi[0] - mj[0])
                b = bucket_for(d)
                if b is None:
                    continue
                counts[b][1] += 1
                if mention_to_pred.get(mi) is not None and mention_to_pred.get(mi) == mention_to_pred.get(mj):
                    counts[b][0] += 1

    # precision: iterate predicted pairs, restricted to mentions that are valid gold mentions
    for c in pred_clusters:
        gold_members = [m for m in c if m in mention_to_gold]
        for i in range(len(gold_members)):
            for j in range(i + 1, len(gold_members)):
                mi, mj = gold_members[i], gold_members[j]
                d = abs(mi[0] - mj[0])
                b = bucket_for(d)
                if b is None:
                    continue
                counts[b][3] += 1
                if mention_to_gold[mi] == mention_to_gold[mj]:
                    counts[b][2] += 1
    return counts


def evaluate_docs(model, hf_split, doc_indices, baseline, tokenizer, device, with_buckets=False):
    model.eval()
    doc_scores = []
    bucket_totals = {name: [0, 0, 0, 0] for name, _, _ in BUCKETS}
    for di in doc_indices:
        ex = hf_split[di]
        gold = gold_clusters_global(ex["mention_clusters"])
        pred = predict_doc_clusters(model, ex, baseline, tokenizer, device)
        doc_scores.append(score_document(gold, pred))
        if with_buckets:
            bc = bucket_pair_counts(gold, pred)
            for name in bucket_totals:
                for k in range(4):
                    bucket_totals[name][k] += bc[name][k]
    model.train()
    agg = aggregate(doc_scores)
    result = {"agg": agg, "doc_scores": doc_scores}
    if with_buckets:
        bucket_metrics = {}
        for name, (r_num, r_den, p_num, p_den) in bucket_totals.items():
            r = r_num / r_den if r_den else 0.0
            p = p_num / p_den if p_den else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            bucket_metrics[name] = {"precision": p, "recall": r, "f1": f1, "n_gold_pairs": r_den, "n_pred_pairs": p_den}
        result["buckets"] = bucket_metrics
    return result
