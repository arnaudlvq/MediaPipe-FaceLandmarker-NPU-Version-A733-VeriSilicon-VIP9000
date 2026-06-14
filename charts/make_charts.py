"""Generate the README charts from the benchmark/results JSON files.

  charts/fidelity.png            - real data (quantization accuracy, available now)
  charts/latency_npu_vs_cpu.png  - placeholder until the on-device run

Usage:  python charts/make_charts.py
Deps:   matplotlib  (pip install matplotlib)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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


def latency_placeholder() -> None:
    """Drawn now, filled in after the on-device run."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    stages = ["face_detector", "face_landmarks", "face_blendshapes", "full pipeline"]
    x = range(len(stages))
    ax.bar(x, [0] * len(stages), color="#cccccc")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages, fontsize=9, color=INK)
    ax.set_ylabel("latency per frame (ms)", color=INK)
    ax.set_ylim(0, 1)
    ax.set_title(
        "A733 latency: NPU vs CPU  -  measurements landing soon",
        color=INK, fontweight="medium",
    )
    ax.text(
        0.5, 0.5, "ON-DEVICE BENCHMARK PENDING\n(Radxa Cubie A7A, VIP9000 NPU)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=15, color="#9a9a9a", fontweight="medium",
    )
    ax.legend(
        handles=[
            Patch(color="#1d9e75", label="NPU (VIP9000, fp16)"),
            Patch(color="#888780", label="CPU (Cortex-A76, XNNPACK)"),
        ],
        frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13),
    )
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "latency_npu_vs_cpu.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/latency_npu_vs_cpu.png (placeholder)")


if __name__ == "__main__":
    fidelity_chart()
    latency_placeholder()
