"""Loss functions for Solar Ring Memory training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LAMBDA_POS, LAMBDA_SPAWN, LAMBDA_RESOLVE


def compute_loss(
    logits: torch.Tensor,        # (B, T, V) or (T, V)
    role_logits: torch.Tensor,   # (B, T, NUM_ROLES) or (T, NUM_ROLES)
    spawn_logits: torch.Tensor,  # (B, T) or (T,)
    token_ids: torch.Tensor,     # (B, T) or (T,)  — target next tokens
    role_labels: torch.Tensor,   # (B, T) or (T,)
    spawn_labels: torch.Tensor,  # (B, T) or (T,)  float 0/1
    pronoun_mask: torch.Tensor = None,   # (B, T) bool
    pronoun_targets: torch.Tensor = None, # (B, T, D) resolved vectors
    pronoun_preds: torch.Tensor = None,   # (B, T, D) predicted vectors
) -> dict:
    """
    total = L_task + 0.3*L_pos + 0.2*L_spawn + 0.2*L_resolve

    L_task   = cross entropy next-token prediction
    L_pos    = cross entropy POS/role labels
    L_spawn  = binary cross entropy conjunction detection
    L_resolve= cosine distance pronoun resolution
    """
    # Flatten batch dims
    if logits.dim() == 3:
        B, T, V = logits.shape
        logits_flat      = logits[:, :-1].reshape(-1, V)          # predict next
        targets_flat     = token_ids[:, 1:].reshape(-1)
        role_flat        = role_logits[:, :-1].reshape(-1, role_logits.shape[-1])
        role_tgt_flat    = role_labels[:, :-1].reshape(-1)
        spawn_flat       = spawn_logits[:, :-1].reshape(-1)
        spawn_tgt_flat   = spawn_labels[:, :-1].reshape(-1).float()
    else:
        T, V = logits.shape
        logits_flat      = logits[:-1]
        targets_flat     = token_ids[1:]
        role_flat        = role_logits[:-1]
        role_tgt_flat    = role_labels[:-1]
        spawn_flat       = spawn_logits[:-1]
        spawn_tgt_flat   = spawn_labels[:-1].float()

    # L_task: next-token cross entropy
    L_task = F.cross_entropy(logits_flat.float(), targets_flat)

    # L_pos: POS/role classification
    L_pos = F.cross_entropy(role_flat.float(), role_tgt_flat.long())

    # L_spawn: binary cross entropy
    L_spawn = F.binary_cross_entropy_with_logits(
        spawn_flat.float(), spawn_tgt_flat
    )

    # L_resolve: cosine distance for pronoun resolution
    L_resolve = torch.tensor(0.0, device=logits.device)
    if (pronoun_mask is not None
            and pronoun_targets is not None
            and pronoun_preds is not None):
        mask = pronoun_mask.bool()
        if mask.any():
            preds   = pronoun_preds[mask]    # (k, d)
            targets = pronoun_targets[mask]  # (k, d)
            # cosine distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(preds.float(), targets.float(), dim=-1)
            L_resolve = (1.0 - cos_sim).mean()

    total = L_task + LAMBDA_POS * L_pos + LAMBDA_SPAWN * L_spawn + LAMBDA_RESOLVE * L_resolve

    return {
        "total":    total,
        "L_task":   L_task,
        "L_pos":    L_pos,
        "L_spawn":  L_spawn,
        "L_resolve": L_resolve,
    }
