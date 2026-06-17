"""Shared mention detector + antecedent linker, fine-tuned on roberta-base.

Architecture (Lee et al.-style span-ranking coreference, simplified):
  1. Encode chunk with RoBERTa (fine-tuned).
  2. Mean-pool subwords -> word-level hidden states.
  3. Enumerate all candidate spans up to `max_span_width` words.
  4. Score each span as a mention candidate (span-based mention detector,
     supports nested/overlapping spans -- unlike BIO tagging).
  5. Prune to the top `prune_ratio * num_words` spans by mention score.
  6. Score every (mention, earlier-mention-or-dummy) pair (antecedent
     linker) and train with the standard marginal log-likelihood loss.

The same architecture/code is reused for all three context-window
baselines; only the chunk boundaries fed in differ (see chunking.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

MAX_SPAN_WIDTH = 10
PRUNE_RATIO = 0.4
WIDTH_BUCKETS = MAX_SPAN_WIDTH  # 1..10
DIST_BUCKET_EDGES = [1, 2, 3, 4, 5, 8, 16, 32, 64]  # 10 buckets total


def distance_bucket(d: torch.Tensor) -> torch.Tensor:
    b = torch.zeros_like(d)
    for edge in DIST_BUCKET_EDGES:
        b += (d >= edge).long()
    return b.clamp(max=len(DIST_BUCKET_EDGES))


class CorefModel(nn.Module):
    def __init__(self, encoder_name: str = "roberta-base", dropout: float = 0.3, span_proj_dim: int = 256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        H = self.encoder.config.hidden_size
        width_dim = 20
        dist_dim = 20
        self.width_emb = nn.Embedding(WIDTH_BUCKETS, width_dim)
        self.dist_emb = nn.Embedding(len(DIST_BUCKET_EDGES) + 1, dist_dim)
        self.span_attn = nn.Linear(H, 1)
        span_repr_dim = 3 * H + width_dim
        self.mention_ffn = nn.Sequential(
            nn.Linear(span_repr_dim, 150), nn.ReLU(), nn.Dropout(dropout), nn.Linear(150, 1)
        )
        self.span_proj = nn.Linear(span_repr_dim, span_proj_dim)
        self.pair_ffn = nn.Sequential(
            nn.Linear(3 * span_proj_dim + dist_dim, 150), nn.ReLU(), nn.Dropout(dropout), nn.Linear(150, 1)
        )
        self.dummy_score = nn.Parameter(torch.zeros(1))

    def pool_words(self, hidden: torch.Tensor, word_ids: list[int | None]) -> torch.Tensor:
        """hidden: [T,H] subword states for ONE example -> [num_words,H]."""
        H = hidden.size(-1)
        num_words = max(w for w in word_ids if w is not None) + 1
        sums = hidden.new_zeros(num_words, H)
        counts = hidden.new_zeros(num_words, 1)
        idx = torch.tensor([w if w is not None else -1 for w in word_ids], device=hidden.device)
        valid = idx >= 0
        sums.index_add_(0, idx[valid], hidden[valid])
        counts.index_add_(0, idx[valid], torch.ones(valid.sum(), 1, device=hidden.device))
        return sums / counts.clamp(min=1)

    def enumerate_spans(self, num_words: int, device) -> torch.Tensor:
        """Returns [num_spans,2] (start,end_inclusive) word indices."""
        spans = []
        for width in range(1, MAX_SPAN_WIDTH + 1):
            if width > num_words:
                break
            s = torch.arange(0, num_words - width + 1, device=device)
            e = s + width - 1
            spans.append(torch.stack([s, e], dim=1))
        return torch.cat(spans, dim=0)

    def span_representations(self, word_hidden: torch.Tensor, spans: torch.Tensor):
        """word_hidden: [num_words,H], spans: [S,2] -> span_repr [S, 3H+width_dim]."""
        num_words, H = word_hidden.shape
        device = word_hidden.device
        s_idx, e_idx = spans[:, 0], spans[:, 1]
        start_emb = word_hidden[s_idx]
        end_emb = word_hidden[e_idx]
        widths = e_idx - s_idx  # 0-indexed width-1
        offsets = torch.arange(MAX_SPAN_WIDTH, device=device)
        gather_idx = (s_idx.unsqueeze(1) + offsets.unsqueeze(0)).clamp(max=num_words - 1)  # [S,W]
        valid_mask = offsets.unsqueeze(0) <= widths.unsqueeze(1)  # [S,W]
        gathered = word_hidden[gather_idx]  # [S,W,H]
        attn_scores = self.span_attn(gathered).squeeze(-1)  # [S,W]
        attn_scores = attn_scores.masked_fill(~valid_mask, float("-inf"))
        attn_w = F.softmax(attn_scores, dim=1)
        attended = (attn_w.unsqueeze(-1) * gathered).sum(1)  # [S,H]
        width_feat = self.width_emb(widths.clamp(max=WIDTH_BUCKETS - 1))
        return torch.cat([start_emb, end_emb, attended, width_feat], dim=-1)

    def forward_one(self, hidden: torch.Tensor, word_ids: list[int | None]):
        """hidden: [T,H] subword states (already sliced for one example)."""
        word_hidden = self.pool_words(hidden, word_ids)
        num_words = word_hidden.size(0)
        spans = self.enumerate_spans(num_words, hidden.device)
        span_repr = self.span_representations(word_hidden, spans)
        mention_scores = self.mention_ffn(span_repr).squeeze(-1)
        g = self.span_proj(span_repr)
        return spans, mention_scores, g

    def forward_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, word_ids_batch: list[list[int | None]]):
        """Batched encoder call, then per-item coreference head. Returns a
        list of (spans, mention_scores, g) tuples, one per batch item."""
        out = self.encoder(input_ids, attention_mask)
        hidden_all = out.last_hidden_state  # [B,T,H]
        results = []
        for b, word_ids in enumerate(word_ids_batch):
            results.append(self.forward_one(hidden_all[b], word_ids))
        return results

    def pairwise_scores(self, g_pruned: torch.Tensor, mention_scores_pruned: torch.Tensor, starts_pruned: torch.Tensor):
        """g_pruned: [M,P], mention_scores_pruned:[M], starts_pruned:[M] (word
        start idx, used for distance buckets, ordered by document position).
        Returns full_scores [M, M+1] where column M is the dummy antecedent,
        and column j<M is "link to pruned-mention j" (only valid for j<i,
        else masked to -inf)."""
        M = g_pruned.size(0)
        device = g_pruned.device
        gi = g_pruned.unsqueeze(1).expand(M, M, -1)
        gj = g_pruned.unsqueeze(0).expand(M, M, -1)
        dist = (starts_pruned.unsqueeze(1) - starts_pruned.unsqueeze(0)).clamp(min=0)
        dist_feat = self.dist_emb(distance_bucket(dist))
        pair_repr = torch.cat([gi, gj, gi * gj, dist_feat], dim=-1)
        pair_scores = self.pair_ffn(pair_repr).squeeze(-1)  # [M,M] pair_scores[i,j]
        total = pair_scores + mention_scores_pruned.unsqueeze(1) + mention_scores_pruned.unsqueeze(0)
        causal_mask = torch.tril(torch.ones(M, M, dtype=torch.bool, device=device), diagonal=-1)
        total = total.masked_fill(~causal_mask, float("-inf"))
        dummy_col = self.dummy_score.expand(M, 1)
        return torch.cat([total, dummy_col], dim=1)  # [M, M+1]


def prune_topk(mention_scores: torch.Tensor, spans: torch.Tensor, num_words: int, prune_ratio: float = PRUNE_RATIO):
    k = max(1, int(prune_ratio * num_words))
    k = min(k, mention_scores.size(0))
    # stable sort by (start position) as tiebreak is handled later; here just topk by score
    top_scores, top_idx = torch.topk(mention_scores, k)
    # reorder by document position (span start, then end) so causal masking is valid
    starts = spans[top_idx, 0]
    ends = spans[top_idx, 1]
    order = torch.argsort(starts * 10000 + ends)
    top_idx = top_idx[order]
    return top_idx
