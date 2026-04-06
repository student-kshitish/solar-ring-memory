"""SolarRingModel: full language model with Solar Ring Memory."""

import torch
import torch.nn as nn

from .config import (
    D_MODEL, N_LAYERS, VOCAB_SIZE, FLAT_SIZE, MAX_SEQ_LEN,
    ROLE_SUBJ, ROLE_OBJ, ROLE_VERB
)

_LAYER_DROPOUT = 0.3
from .layers import SolarRingLayer
from .solar_memory import SolarMemory


class SolarRingModel(nn.Module):
    """
    Solar Ring Memory language model.

    Architecture:
        token_ids → Embedding
        For each token (sequentially):
            Pass through 8 SolarRingLayers, updating one shared SolarMemory
            W_skip residual from layer-1 output to layer-8 input
        Flatten memory → W_out → logits
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, pretrained_embeddings=None):
        super().__init__()
        self.vocab_size = vocab_size
        d = D_MODEL

        # Token embedding (optionally initialized from pretrained GloVe)
        self.embedding = nn.Embedding(vocab_size, d)
        if pretrained_embeddings is not None:
            import torch
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = False  # freeze; caller unfreezes

        # 8 stacked SolarRingLayers
        self.layers = nn.ModuleList([SolarRingLayer(i) for i in range(N_LAYERS)])

        # Dropout applied between consecutive layers (not after the final layer)
        self.layer_dropout = nn.Dropout(p=_LAYER_DROPOUT)

        # Skip connection: project layer-1 output to layer-8 input space
        self.W_skip = nn.Linear(d, d, bias=False)

        # Flatten projection: 13*8*512 → 512
        self.W_out = nn.Linear(FLAT_SIZE, d)
        self.out_norm = nn.LayerNorm(d)

        # Language model head
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

        # Tie embedding and lm_head weights
        self.lm_head.weight = self.embedding.weight

    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,      # (B, T) or (T,)
        role_labels: torch.Tensor = None,   # (B, T) or (T,) int
        spawn_labels: torch.Tensor = None,  # (B, T) or (T,) float
        pronoun_mask: torch.Tensor = None,  # (B, T) or (T,) bool
    ):
        """
        Process a sequence token by token through the ring memory.

        Returns:
            logits:       (B, T, V) next-token logits
            aux_outputs:  dict with role_logits, spawn_logits per token
        """
        squeeze = False
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            if role_labels is not None:
                role_labels = role_labels.unsqueeze(0)
            if spawn_labels is not None:
                spawn_labels = spawn_labels.unsqueeze(0)
            if pronoun_mask is not None:
                pronoun_mask = pronoun_mask.unsqueeze(0)
            squeeze = True

        B, T = token_ids.shape
        device = token_ids.device
        dtype = next(self.parameters()).dtype

        all_logits = []
        all_role_logits = []
        all_spawn_logits = []
        all_context_vecs = []

        for b in range(B):
            # hard_lock=True at inference so subject/object poles are immutable;
            # hard_lock=False during training enables soft refinement.
            memory = SolarMemory(device=device, dtype=dtype,
                                 hard_lock=not self.training)

            seq_logits = []
            seq_role_logits = []
            seq_spawn_logits = []

            for t in range(T):
                tok = token_ids[b, t]
                x = self.embedding(tok).to(dtype)  # (d,)

                role_gt = role_labels[b, t].item() if role_labels is not None else None
                is_pron = pronoun_mask[b, t].item() if pronoun_mask is not None else False

                skip_x = None
                t_role_logits = []
                t_spawn_logits = []

                for layer_idx, layer in enumerate(self.layers):
                    x, role_l, spawn_l = layer(
                        x, memory,
                        role_label=role_gt,
                        is_pronoun=bool(is_pron),
                        write_enabled=(layer_idx == 0),
                    )
                    t_role_logits.append(role_l)
                    t_spawn_logits.append(spawn_l)

                    # Save L1 output for skip connection
                    if layer_idx == 0:
                        skip_x = x

                    # Dropout between layers (not after the last layer)
                    if layer_idx < N_LAYERS - 1:
                        x = self.layer_dropout(x)

                # Add skip residual before final representation
                x = x + self.W_skip(skip_x.float()).to(dtype)

                seq_logits.append(x)
                # Average role / spawn logits across layers for supervision
                seq_role_logits.append(torch.stack(t_role_logits).mean(0))
                seq_spawn_logits.append(torch.stack(t_spawn_logits).mean(0))

            # Flatten memory to fixed representation
            flat = memory.flatten()                          # (FLAT_SIZE,)
            mem_vec = self.out_norm(self.W_out(flat.float()).to(dtype))  # (d,)
            all_context_vecs.append(mem_vec)                 # for classification head

            # Combine per-token representations with memory context
            seq_tensor = torch.stack(seq_logits, dim=0)     # (T, d)
            seq_tensor = seq_tensor + mem_vec.unsqueeze(0)  # broadcast

            logits = self.lm_head(seq_tensor.float()).to(dtype)  # (T, V)
            all_logits.append(logits)
            all_role_logits.append(torch.stack(seq_role_logits))   # (T, NUM_ROLES)
            all_spawn_logits.append(torch.stack(seq_spawn_logits)) # (T,)

        logits_out  = torch.stack(all_logits, dim=0)          # (B, T, V)
        role_out    = torch.stack(all_role_logits, dim=0)     # (B, T, NUM_ROLES)
        spawn_out   = torch.stack(all_spawn_logits, dim=0)    # (B, T)
        context_out = torch.stack(all_context_vecs, dim=0)    # (B, d)

        if squeeze:
            logits_out  = logits_out.squeeze(0)
            role_out    = role_out.squeeze(0)
            spawn_out   = spawn_out.squeeze(0)
            context_out = context_out.squeeze(0)               # (d,)

        aux = {"role_logits": role_out, "spawn_logits": spawn_out,
               "context_vec": context_out}
        return logits_out, aux

    def run_with_physics(
        self,
        token_ids: torch.Tensor,   # (T,) 1-D sequence
        words: list,               # list of str, len == T
        manager,                   # BlackWhiteHoleManager (pre-created by caller)
        sun_state,                 # SunState (pre-created by caller)
    ):
        """
        Physics-enhanced forward pass.
        Additive — existing forward() is unchanged.

        Returns:
            logits:      (T, V) next-token logits
            context_vec: (d,)  flattened memory representation
        """
        device = token_ids.device
        dtype  = next(self.parameters()).dtype

        if token_ids.dim() == 2:
            token_ids = token_ids.squeeze(0)
        T = token_ids.shape[0]

        memory = SolarMemory(device=device, dtype=dtype,
                             hard_lock=not self.training)

        seq_reps = []
        skip_x   = None

        for t in range(T):
            tok       = token_ids[t]
            x         = self.embedding(tok).to(dtype)
            word      = words[t] if t < len(words) else ''

            for layer_idx, layer in enumerate(self.layers):
                x, _, _ = layer.forward_with_physics(
                    x, memory,
                    token_text=word,
                    token_pos=t,
                    manager=manager,
                    sun_state=sun_state,
                )
                if layer_idx == 0:
                    skip_x = x
                if layer_idx < N_LAYERS - 1:
                    x = self.layer_dropout(x)

            x = x + self.W_skip(skip_x.float()).to(dtype)
            seq_reps.append(x)

        flat        = memory.flatten()
        context_vec = self.out_norm(self.W_out(flat.float()).to(dtype))

        seq_tensor  = torch.stack(seq_reps, dim=0)        # (T, d)
        seq_tensor  = seq_tensor + context_vec.unsqueeze(0)
        logits      = self.lm_head(seq_tensor.float()).to(dtype)  # (T, V)

        return logits, context_vec

    def get_memory_for_sentence(self, token_ids: torch.Tensor,
                                role_labels: torch.Tensor = None,
                                spawn_labels: torch.Tensor = None,
                                pronoun_mask: torch.Tensor = None) -> SolarMemory:
        """
        Run forward pass and return the final SolarMemory (for inspection).
        Works on a single sequence (1D or (1,T) tensor).
        """
        device = token_ids.device
        dtype = next(self.parameters()).dtype
        device_type = "cuda" if device.type == "cuda" else "cpu"

        if token_ids.dim() == 2:
            token_ids = token_ids.squeeze(0)
        T = token_ids.shape[0]

        memory = SolarMemory(device=device, dtype=dtype, hard_lock=not self.training)

        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            for t in range(T):
                tok = token_ids[t]
                x = self.embedding(tok).to(dtype)

                role_gt = role_labels[t].item() if role_labels is not None else None
                is_pron = pronoun_mask[t].item() if pronoun_mask is not None else False

                skip_x = None
                for layer_idx, layer in enumerate(self.layers):
                    x, _, _ = layer(x, memory, role_label=role_gt,
                                    is_pronoun=bool(is_pron),
                                    write_enabled=(layer_idx == 0))
                    if layer_idx == 0:
                        skip_x = x

                x = x + self.W_skip(skip_x.float()).to(dtype)

        return memory
