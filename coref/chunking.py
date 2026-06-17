"""Splits a PreCo document's sentences into non-overlapping context chunks.

All three baselines share one mechanism: tile the document into
chunks of whole sentences. Only mentions that fall in the same chunk
can be linked to each other -- this is the single independent
variable across baselines (a)/(b)/(c). Every token is encoded exactly
once per document regardless of baseline, so training cost is
comparable across the three.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BaselineName = Literal["sentence", "window5", "fulldoc"]


@dataclass
class Chunk:
    sent_range: tuple[int, int]  # [start, end) sentence indices in the doc
    words: list[str]
    # word_doc_id[local_word_idx] = (sent_idx, intra_sent_idx) global address
    word_doc_id: list[tuple[int, int]] = field(default_factory=list)


def make_chunks(sentences: list[list[str]], baseline: BaselineName, max_tokens_estimator) -> list[Chunk]:
    """max_tokens_estimator(words: list[str]) -> int subword-token count
    (used only by the 'fulldoc' baseline to respect the 512 cap)."""
    n = len(sentences)
    if baseline == "sentence":
        groups = [[i] for i in range(n)]
    elif baseline == "window5":
        groups = [list(range(i, min(i + 5, n))) for i in range(0, n, 5)]
    elif baseline == "fulldoc":
        groups = []
        cur: list[int] = []
        cur_tokens = 0
        for i in range(n):
            sent_tokens = max_tokens_estimator(sentences[i])
            # +2 for BOS/EOS overhead margin
            if cur and cur_tokens + sent_tokens + 2 > 512:
                groups.append(cur)
                cur = []
                cur_tokens = 0
            cur.append(i)
            cur_tokens += sent_tokens
        if cur:
            groups.append(cur)
    else:
        raise ValueError(baseline)

    chunks = []
    for g in groups:
        words: list[str] = []
        word_doc_id: list[tuple[int, int]] = []
        for sent_idx in g:
            for w_idx, w in enumerate(sentences[sent_idx]):
                words.append(w)
                word_doc_id.append((sent_idx, w_idx))
        chunks.append(Chunk(sent_range=(g[0], g[-1] + 1), words=words, word_doc_id=word_doc_id))
    return chunks


def mentions_to_chunk_local(mention_clusters: list[list[list[int]]], chunks: list[Chunk]):
    """For each chunk, return gold_clusters_local: list of clusters, each a
    list of local (start,end_inclusive) word spans -- only mentions whose
    sentence falls inside that chunk are included (mentions never cross
    chunk boundaries since chunks are whole sentences)."""
    # map (sent_idx) -> chunk_idx
    sent_to_chunk = {}
    for ci, c in enumerate(chunks):
        for s in range(c.sent_range[0], c.sent_range[1]):
            sent_to_chunk[s] = ci
    # map (sent_idx, intra_idx) -> local word idx within its chunk
    local_idx = {}
    for c in chunks:
        for li, (s, w) in enumerate(c.word_doc_id):
            local_idx[(s, w)] = li

    per_chunk_clusters: list[list[list[tuple[int, int]]]] = [[] for _ in chunks]
    for cluster in mention_clusters:
        by_chunk: dict[int, list[tuple[int, int]]] = {}
        for sent_idx, start, end in cluster:
            if sent_idx not in sent_to_chunk:
                continue
            ci = sent_to_chunk[sent_idx]
            ls = local_idx[(sent_idx, start)]
            le = local_idx[(sent_idx, end - 1)]
            by_chunk.setdefault(ci, []).append((ls, le))
        for ci, spans in by_chunk.items():
            per_chunk_clusters[ci].append(spans)
    return per_chunk_clusters
