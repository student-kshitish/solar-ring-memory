import torch
import torch.nn as nn
from solar_ring.contextual_embedder import ContextualEmbedder
from solar_ring.layers import SolarRingLayer
from solar_ring.solar_memory import SolarMemory

class SolarRingContextual(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.d_in = 384   # MiniLM output dim
        self.d = 300      # SolarRingLayer internal dim (D_MODEL from config)
        self.device = device
        self.embedder = ContextualEmbedder(device)
        self.input_proj = nn.Linear(384, 300)   # project MiniLM → ring space
        self.layers = nn.ModuleList([
            SolarRingLayer(layer_idx=i)
            for i in range(8)
        ])
        self.W_skip = nn.Linear(300, 384)       # skip: ring space → output space
        flat_dim = 13 * 8 * 300                 # 31200
        self.W_out = nn.Linear(flat_dim, 384)
        self.out_norm = nn.LayerNorm(384)
        self.pronoun_head = nn.Linear(384, 1)
        self.small_head = nn.Sequential(   # ~25K params
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.to(device)

    def forward(self, sentence: str, memory=None):
        word_embeddings = self.embedder.embed_words(sentence)
        word_embeddings = self.input_proj(word_embeddings)
        if memory is None:
            memory = SolarMemory(device=self.device)
        x0 = word_embeddings[0]
        for t in range(word_embeddings.shape[0]):
            h_t = word_embeddings[t]
            for layer_idx, layer in enumerate(self.layers):
                h_t, r_t, sp = layer(
                    h_t, memory,
                    write_enabled=(layer_idx == 0)
                )
        flat = memory.flatten()
        flat_dim = 13 * 8 * 300
        if flat.shape[0] < flat_dim:
            flat = torch.cat([
                flat,
                torch.zeros(
                    flat_dim - flat.shape[0],
                    device=self.device
                )
            ])
        flat = flat[:flat_dim].float()
        c = self.W_out(flat)
        c = self.out_norm(c + self.W_skip(x0))
        logit = self.pronoun_head(c)
        return c, memory, logit

    def forward_from_emb(self, word_embeddings: torch.Tensor, memory=None):
        """Forward pass using pre-cached MiniLM embeddings (seq_len, 384)."""
        word_embeddings = self.input_proj(word_embeddings)
        if memory is None:
            memory = SolarMemory(device=self.device)
        x0 = word_embeddings[0]
        for t in range(word_embeddings.shape[0]):
            h_t = word_embeddings[t]
            for layer_idx, layer in enumerate(self.layers):
                h_t, r_t, sp = layer(
                    h_t, memory,
                    write_enabled=(layer_idx == 0)
                )
        flat = memory.flatten()
        flat_dim = 13 * 8 * 300
        if flat.shape[0] < flat_dim:
            flat = torch.cat([
                flat,
                torch.zeros(flat_dim - flat.shape[0], device=self.device)
            ])
        flat = flat[:flat_dim].float()
        c = self.W_out(flat)
        c = self.out_norm(c + self.W_skip(x0))
        logit = self.small_head(c)
        return c, memory, logit

    def freeze_ring_layers(self):
        """Freeze all layers except output heads (W_out, out_norm, pronoun_head)."""
        for param in self.layers.parameters():
            param.requires_grad = False
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.W_skip.parameters():
            param.requires_grad = False

    def freeze_for_probe(self):
        """Freeze everything except small_head (~25K params)."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.small_head.parameters():
            param.requires_grad = True

    def forward_mean(self, word_embeddings: torch.Tensor):
        """
        Simple forward using mean of word embeddings.
        Same path used in both training and evaluation.
        word_embeddings: (L, 384) cached tensor
        """
        x = self.input_proj(word_embeddings.clone())   # (L, 300)
        mean_vec = x.mean(dim=0)                        # (300,)
        c = self.out_norm(self.W_skip(mean_vec))        # (384,)
        logit = self.pronoun_head(c)
        return c, logit

    def count_parameters(self):
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )
