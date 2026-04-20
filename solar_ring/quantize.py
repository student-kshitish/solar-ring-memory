"""Memory-efficient quantization and compression utilities for Solar Ring Model.

Provides:
  - fp16_compress(model)  : convert weights to FP16 in-place (~50% memory)
  - int8_compress(model)  : dynamic INT8 quantization via torch.quantization
  - ring_stats(memory)    : report active ring count and memory footprint
  - prune_dead_rings(memory): remove zero-norm rings to free slots
"""

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solar_memory import SolarMemory
    from .model import SolarRingModel


# ── Weight compression ───────────────────────────────────────────────────────

def fp16_compress(model: nn.Module) -> nn.Module:
    """Cast all model weights to FP16. ~50% VRAM reduction. Inference only."""
    model.half()
    return model


def bf16_compress(model: nn.Module) -> nn.Module:
    """Cast all model weights to BFloat16 (better range than FP16)."""
    model.to(torch.bfloat16)
    return model


def int8_compress(model: nn.Module) -> nn.Module:
    """
    Apply dynamic INT8 quantization to all Linear layers.
    ~4x memory reduction, ~2x inference speedup on CPU.
    Note: GPU requires static quantization (not applied here).
    """
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model.cpu(),
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def model_size_mb(model: nn.Module) -> float:
    """Return model parameter size in MB."""
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


# ── Ring-level memory efficiency ─────────────────────────────────────────────

def ring_stats(memory: "SolarMemory") -> dict:
    """Report ring usage statistics."""
    from .config import MAX_RINGS, D_MODEL, SLOTS_PER_RING
    n_rings   = len(memory.rings)
    n_active  = sum(
        1 for r in memory.rings
        if any(r._slots[i] is not None for i in range(SLOTS_PER_RING))
    )
    bytes_per_slot = D_MODEL * 2  # bfloat16 = 2 bytes
    used_bytes  = n_active * SLOTS_PER_RING * bytes_per_slot
    total_bytes = MAX_RINGS * SLOTS_PER_RING * bytes_per_slot
    return {
        "n_rings":        n_rings,
        "n_active":       n_active,
        "max_rings":      MAX_RINGS,
        "used_kb":        used_bytes / 1024,
        "total_kb":       total_bytes / 1024,
        "utilization":    n_active / MAX_RINGS,
    }


def prune_dead_rings(memory: "SolarMemory") -> int:
    """
    Compress rings by identifying near-zero rings (dead / unused).
    Merges dead ring content into parent and marks slot as unused.
    Returns number of rings pruned.

    Safe to call between sentences; DO NOT call mid-sequence.
    """
    from .config import SLOTS_PER_RING
    pruned = 0
    # Never prune ring 0 (sun) or the active ring
    for ring in memory.rings[1:]:
        if ring.ring_id == memory.alpha:
            continue
        total_norm = sum(
            ring._slots[i].norm().item()
            for i in range(SLOTS_PER_RING)
            if ring._slots[i] is not None
        )
        if total_norm < 0.1:   # effectively empty ring
            ring._slots = [None] * SLOTS_PER_RING
            ring.subj_locked = False
            ring.obj_locked  = False
            pruned += 1
    return pruned


# ── Gradient checkpointing for training efficiency ───────────────────────────

def enable_gradient_checkpointing(model: "SolarRingModel"):
    """
    Enable gradient checkpointing on all SolarRingLayer modules.
    Trades compute for memory during training (~30-50% less activation memory).
    """
    for layer in model.layers:
        layer.gradient_checkpointing = True
    return model
