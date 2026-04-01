"""Run all benchmarks and print a comparison table.

Usage:
    python benchmarks/run_all.py

Output:
    Model        | Winograd Acc | Parameters  | Memory MB
    Solar Ring   |       ??%    |     ??M     |    ??
    Vanilla LSTM |       ??%    |     ??M     |    ??
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from benchmarks.winograd import run_winograd_benchmark

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_memory_mb(model: torch.nn.Module) -> float:
    """Estimate memory footprint of model parameters in MB (bfloat16 = 2 bytes)."""
    n_params = count_parameters(model)
    return (n_params * 2) / (1024 ** 2)


def print_table(rows: list, headers: list) -> None:
    col_widths = [max(len(h), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in col_widths)
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def main():
    print("=" * 60)
    print("Solar Ring Memory — Full Benchmark Suite")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Winograd schema benchmark (loads both models internally)
    # ----------------------------------------------------------------
    print("\n[1/1] Winograd Schema Challenge")
    results = run_winograd_benchmark()

    solar_model = results["solar_model"]
    vanilla_model = results["vanilla_model"]

    solar_params = count_parameters(solar_model)
    vanilla_params = count_parameters(vanilla_model)
    solar_mem = model_memory_mb(solar_model)
    vanilla_mem = model_memory_mb(vanilla_model)

    solar_winograd = results["solar_winograd"]
    vanilla_winograd = results["vanilla_winograd"]

    # ----------------------------------------------------------------
    # Comparison table
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Comparison Table")
    print("=" * 60)

    headers = ["Model", "Winograd Acc", "Parameters", "Memory MB"]
    rows = [
        [
            "Solar Ring",
            f"{solar_winograd:.1%}",
            f"{solar_params / 1e6:.1f}M",
            f"{solar_mem:.1f}",
        ],
        [
            "Vanilla LSTM",
            f"{vanilla_winograd:.1%}",
            f"{vanilla_params / 1e6:.1f}M",
            f"{vanilla_mem:.1f}",
        ],
    ]
    print_table(rows, headers)

    # Delta summary
    delta_acc = solar_winograd - vanilla_winograd
    sign = "+" if delta_acc >= 0 else ""
    print(f"\nSolar Ring vs Vanilla LSTM  Δ Winograd: {sign}{delta_acc:.1%}")
    print(f"Parameter ratio: {solar_params / vanilla_params:.2f}x")


if __name__ == "__main__":
    main()
