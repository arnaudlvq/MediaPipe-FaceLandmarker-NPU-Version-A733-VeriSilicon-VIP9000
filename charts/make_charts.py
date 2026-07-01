"""Generate the README charts from the benchmark/results JSON files.

  charts/fidelity.png            - quantization accuracy vs reference TFLite
  charts/latency_npu_vs_cpu.png  - measured NPU (int16/fp16) vs CPU latency

Usage:  python charts/make_charts.py
Deps:   matplotlib  (pip install matplotlib)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
OUT = Path(__file__).resolve().parent

INK = "#1b1b1b"
GRID = "#d8d8d8"
COLORS = {
    "fp16": "#1d9e75",      # the recommended path - lossless
    "bf16": "#378add",
    "int16": "#ba7517",
    "pcq_int8": "#e24b4a",  # rejected - degrades landmarks
}
LABELS = {"fp16": "FP16", "bf16": "BF16", "int16": "INT16", "pcq_int8": "INT8 (pcq)"}


def fidelity_chart() -> None:
    data = json.loads((RESULTS / "fidelity.json").read_text())
    models = data["models"]
    names = list(models)
    precisions = ["fp16", "bf16", "int16", "pcq_int8"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    n = len(precisions)
    width = 0.8 / n
    for i, prec in enumerate(precisions):
        vals = [max(models[m][prec], 1e-5) for m in names]
        xs = [j + (i - n / 2) * width + width / 2 for j in range(len(names))]
        ax.bar(xs, vals, width, label=LABELS[prec], color=COLORS[prec])

    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{m}\n({models[m]['unit']})" for m in names], fontsize=9, color=INK
    )
    ax.set_ylabel("mean abs error vs TFLite  (log scale, lower = better)", color=INK)
    ax.set_title(
        "Conversion fidelity per precision  -  FP16 is lossless, INT8 is not",
        color=INK, fontweight="medium",
    )
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fidelity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/fidelity.png")


def latency_chart() -> None:
    """Measured on the Radxa Cubie A7A (A733): NPU (int16 & fp16) vs CPU."""
    data = json.loads((RESULTS / "latency.json").read_text())
    models = ["face_detector", "face_landmarks_detector", "face_blendshapes"]
    labels = ["face_detector", "face_landmarks", "face_blendshapes"]
    series = [
        ("NPU int16", "#1d9e75", [data["npu_int16"][m] for m in models]),
        ("NPU fp16", "#e24b4a", [data["npu_fp16"][m] for m in models]),
    ]
    cpu = data["cpu_mediapipe_end_to_end_ms"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    n = len(series)
    width = 0.8 / n
    for i, (name, colour, vals) in enumerate(series):
        xs = [j + (i - n / 2) * width + width / 2 for j in range(len(models))]
        bars = ax.bar(xs, vals, width, label=name, color=colour)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=INK)

    ax.axhline(cpu, color="#888780", linewidth=1.4, linestyle="--")
    ax.text(len(models) - 0.5, cpu * 1.05, f"CPU full frame  {cpu:.0f} ms",
            ha="right", va="bottom", color="#5f5e5a", fontsize=9)

    ax.set_yscale("log")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel("inference latency (ms, log scale)", color=INK)
    ax.set_title(
        "A733: NPU int16 is ~8x faster than CPU  -  fp16 is a trap (30x slower than int16)",
        color=INK, fontweight="medium", fontsize=11,
    )
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "latency_npu_vs_cpu.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/latency_npu_vs_cpu.png")


if __name__ == "__main__":
    fidelity_chart()
    latency_chart()
