"""Vanilla LSTM baseline — parameter-matched to SolarRingModel.

Two-layer LSTM with d_model=512, same embedding and lm_head as
SolarRingModel. Used as a comparison baseline in benchmarks.
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solar_ring.config import D_MODEL, VOCAB_SIZE


class VanillaLSTM(nn.Module):
    """
    Standard 2-layer LSTM language model.

    Architecture:
        token_ids → Embedding(vocab, d_model)
        → LSTM(d_model, d_model, num_layers=2, batch_first=True)
        → Linear(d_model, vocab)   [lm_head, weights tied to embedding]

    Parameter count is matched to SolarRingModel by setting hidden_size=512
    and num_layers=2. The lm_head is identical (tied embedding weights).
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, pretrained_embeddings=None):
        super().__init__()
        self.vocab_size = vocab_size
        d = D_MODEL

        self.embedding = nn.Embedding(vocab_size, d)
        if pretrained_embeddings is not None:
            import torch
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = False  # freeze; caller unfreezes

        # 2-layer LSTM; proj_size omitted so hidden == d_model
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=d,
            num_layers=2,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(d)

        # Language model head — tied to embedding
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,   # (B, T) or (T,)
        hidden=None,               # optional initial (h_0, c_0)
    ):
        """
        Returns:
            logits:  (B, T, V)  next-token logits
            hidden:  final LSTM hidden state tuple (for stateful use)
        """
        squeeze = False
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze = True

        dtype = next(self.parameters()).dtype
        x = self.embedding(token_ids).to(dtype)          # (B, T, d)
        out, hidden = self.lstm(x, hidden)               # (B, T, d)
        out = self.out_norm(out)
        logits = self.lm_head(out.float()).to(dtype)     # (B, T, V)

        if squeeze:
            logits = logits.squeeze(0)

        return logits, hidden
