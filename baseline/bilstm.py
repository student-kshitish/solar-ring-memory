"""Bidirectional LSTM baseline — parameter-matched comparison model.

2-layer BiLSTM with d_model=300, hidden_size=150 per direction
(total hidden = 300 after concat), same vocab size as SolarRingModel.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solar_ring.config import D_MODEL, VOCAB_SIZE


class BiLSTM(nn.Module):
    """
    Bidirectional 2-layer LSTM language model.

    Architecture:
        token_ids → Embedding(vocab, d_model)
        → BiLSTM(d_model, d_model//2, num_layers=2, bidirectional=True)
          output dim = d_model//2 * 2 = d_model
        → LayerNorm(d_model)
        → Linear(d_model, vocab)   [separate lm_head — not tied]
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, pretrained_embeddings=None):
        super().__init__()
        self.vocab_size = vocab_size
        d = D_MODEL

        self.embedding = nn.Embedding(vocab_size, d)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = False  # freeze; caller unfreezes

        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=d // 2,        # 150 per direction → 300 concatenated
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3,               # applied between layer 1 and 2
        )

        self.norm = nn.LayerNorm(d)

        # Separate lm_head (not weight-tied — BiLSTM is bidirectional at training
        # but scoring at eval uses the causal prefix, so tied weights would be odd)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    # ------------------------------------------------------------------

    def forward(self, token_ids: torch.Tensor, hidden=None):
        """
        Returns:
            logits:  (B, T, V)
            hidden:  final LSTM hidden state
        """
        squeeze = False
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze = True

        dtype = next(self.parameters()).dtype
        x   = self.embedding(token_ids).to(dtype)     # (B, T, d)
        out, hidden = self.lstm(x, hidden)             # (B, T, d)
        out = self.norm(out)
        logits = self.lm_head(out.float()).to(dtype)   # (B, T, V)

        if squeeze:
            logits = logits.squeeze(0)

        return logits, hidden
