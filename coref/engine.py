from __future__ import annotations

import torch
import torch.nn.functional as F

from coref.model import CorefModel, prune_topk


def _span_index_map(spans: torch.Tensor) -> dict[tuple[int, int], int]:
    return {(int(s), int(e)): i for i, (s, e) in enumerate(spans.tolist())}


def compute_chunk_loss(model: CorefModel, spans: torch.Tensor, mention_scores: torch.Tensor, g: torch.Tensor,
                        num_words: int, gold_clusters_local: list[list[tuple[int, int]]]):
    device = mention_scores.device
    span_to_idx = _span_index_map(spans)

    gold_span_set = {sp for cluster in gold_clusters_local for sp in cluster}
    labels = torch.zeros(spans.size(0), device=device)
    for sp in gold_span_set:
        if sp in span_to_idx:
            labels[span_to_idx[sp]] = 1.0
    num_pos = labels.sum().item()
    pos_weight = torch.tensor([(labels.numel() - num_pos) / max(num_pos, 1.0)], device=device) if num_pos > 0 else torch.tensor([1.0], device=device)
    mention_loss = F.binary_cross_entropy_with_logits(mention_scores, labels, pos_weight=pos_weight)

    top_idx = prune_topk(mention_scores, spans, num_words)
    g_p, ms_p, spans_p = g[top_idx], mention_scores[top_idx], spans[top_idx]
    starts_p = spans_p[:, 0]
    M = g_p.size(0)
    pruned_span_to_local = {(int(s), int(e)): i for i, (s, e) in enumerate(spans_p.tolist())}

    span_to_cluster: dict[tuple[int, int], int] = {}
    for ci, cluster in enumerate(gold_clusters_local):
        for sp in cluster:
            span_to_cluster[sp] = ci

    if M == 0:
        return mention_loss

    full_scores = model.pairwise_scores(g_p, ms_p, starts_p)  # [M, M+1]
    logsumexp_all = torch.logsumexp(full_scores, dim=1)  # [M]

    ante_terms = []
    for local_i, (s, e) in enumerate(spans_p.tolist()):
        cid = span_to_cluster.get((s, e))
        if cid is None:
            continue  # not a gold mention -- no antecedent supervision for junk spans
        gold_antecedents = []
        for local_j in range(local_i):
            sj, ej = int(spans_p[local_j, 0]), int(spans_p[local_j, 1])
            if span_to_cluster.get((sj, ej)) == cid:
                gold_antecedents.append(local_j)
        if gold_antecedents:
            cols = torch.tensor(gold_antecedents, device=device)
            target_logits = full_scores[local_i, cols]
        else:
            target_logits = full_scores[local_i, M].unsqueeze(0)  # dummy column
        log_p = torch.logsumexp(target_logits, dim=0) - logsumexp_all[local_i]
        ante_terms.append(-log_p)

    if ante_terms:
        antecedent_loss = torch.stack(ante_terms).mean()
    else:
        antecedent_loss = torch.zeros((), device=device)

    return mention_loss + antecedent_loss


def predict_chunk_clusters(model: CorefModel, spans: torch.Tensor, mention_scores: torch.Tensor, g: torch.Tensor,
                            num_words: int) -> list[list[tuple[int, int]]]:
    top_idx = prune_topk(mention_scores, spans, num_words)
    g_p, ms_p, spans_p = g[top_idx], mention_scores[top_idx], spans[top_idx]
    keep = ms_p > 0
    if keep.sum() == 0:
        return []
    g_k, ms_k, spans_k = g_p[keep], ms_p[keep], spans_p[keep]
    starts_k = spans_k[:, 0]
    M = g_k.size(0)
    full_scores = model.pairwise_scores(g_k, ms_k, starts_k)  # [M, M+1]
    best = full_scores.argmax(dim=1).tolist()  # values in 0..M (M = dummy)

    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, b in enumerate(best):
        if b < M:
            union(i, b)

    groups: dict[int, list[int]] = {}
    for i in range(M):
        groups.setdefault(find(i), []).append(i)

    clusters = []
    for members in groups.values():
        clusters.append([(int(spans_k[m, 0]), int(spans_k[m, 1])) for m in members])
    return clusters


def local_span_to_global_id(word_doc_id: list[tuple[int, int]], start: int, end_incl: int):
    sent_s, w_s = word_doc_id[start]
    sent_e, w_e = word_doc_id[end_incl]
    return (sent_s, w_s, sent_e, w_e + 1)


def global_id_for_gold(sent_idx: int, start: int, end_excl: int):
    return (sent_idx, start, sent_idx, end_excl)
