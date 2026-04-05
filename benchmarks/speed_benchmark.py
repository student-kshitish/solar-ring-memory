"""Benchmark 5: Inference speed across sequence lengths."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import time

from solar_ring.model import SolarRingModel
from baseline.bilstm import BiLSTM

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LENGTHS  = [10, 25, 50, 100, 200, 500]
N_ITER   = 10
VOCAB    = 500


def measure_speed(model, length, n_iter):
    times = []
    ids = torch.randint(0, VOCAB, (length,), device=DEVICE)
    with torch.no_grad():
        for _ in range(n_iter):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(ids)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000  # ms


solar  = SolarRingModel(vocab_size=VOCAB).to(DEVICE)
bilstm = BiLSTM(vocab_size=VOCAB).to(DEVICE)
solar.eval()
bilstm.eval()

print(f"\n{'Length':>8} | {'Solar Ring':>12} | {'BiLSTM':>10} | {'Ratio (BiLSTM/SR)':>18}")
print("-" * 60)

results = {}
for L in LENGTHS:
    sr_ms  = measure_speed(solar,  L, N_ITER)
    bil_ms = measure_speed(bilstm, L, N_ITER)
    ratio  = bil_ms / sr_ms if sr_ms > 0 else 0
    results[L] = (sr_ms, bil_ms, ratio)
    print(f"{L:>8} | {sr_ms:>10.2f}ms | {bil_ms:>8.2f}ms | {ratio:>16.2f}x")

print("-" * 60)
sr_500, bil_500, _ = results[500]
winner = "Solar Ring" if sr_500 < bil_500 else "BiLSTM"
print(f"  Winner at L=500: {winner}")
