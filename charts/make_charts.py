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
    """Measured on the Radxa Cubie A7A (A733): NPU (int16 & fp16) vs CPU, iso-scope."""
    data = json.loads((RESULTS / "latency.json").read_text())
    models = ["face_detector", "face_landmarks_detector", "face_blendshapes"]
    labels = ["face_detector", "face_landmarks", "face_blendshapes"]
    series = [
        ("NPU int16", "#1d9e75", [data["npu_int16"][m] for m in models]),
        ("CPU fp32 (1x A76)", "#888780", [data["cpu_tflite_fp32_1thread"][m] for m in models]),
        ("NPU fp16", "#e24b4a", [data["npu_fp16"][m] for m in models]),
    ]
    cpu_e2e = data["cpu_mediapipe_end_to_end_ms"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    n = len(series)
    width = 0.8 / n
    for i, (name, colour, vals) in enumerate(series):
        xs = [j + (i - n / 2) * width + width / 2 for j in range(len(models))]
        bars = ax.bar(xs, vals, width, label=name, color=colour)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=INK)

    ax.axhline(cpu_e2e, color="#bbb9b2", linewidth=1.2, linestyle="--")
    ax.text(len(models) - 0.5, cpu_e2e * 1.05,
            f"CPU MediaPipe full frame (incl. pre/post)  {cpu_e2e:.0f} ms",
            ha="right", va="bottom", color="#8a8880", fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel("pure inference latency (ms, log scale)", color=INK)
    ax.set_title(
        "A733, iso-scope: NPU int16 is 5.6x faster than CPU  -  fp16 is a trap (29x the cycles)",
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


def energy_chart() -> None:
    """Measured whole-board power (USB meter): the NPU's energy advantage."""
    data = json.loads((RESULTS / "power.json").read_text())
    idle = data["idle_w"]
    at15 = data["at_15fps"]
    order = [("npu_int16", "NPU int16", "#1d9e75"),
             ("cpu_2x_a76", "CPU 2× A76", "#ca6f1e"),
             ("cpu_1x_a76", "CPU 1× A76", "#e59866")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})

    # left: marginal power for the same 15 fps workload
    names = [o[1] for o in order]
    vals = [at15[o[0]]["marginal_w"] for o in order]
    bars = ax.barh(names, vals, color=[o[2] for o in order], height=0.6)
    ax.bar_label(bars, fmt="+%.2f W", padding=3, fontsize=10, color=INK, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("extra board power vs idle (W)", color=INK)
    ax.set_title("Same face pipeline @ 15 fps\nNPU adds ~7× less power", color=INK,
                 fontweight="medium", fontsize=11)
    ax.invert_yaxis()

    # right: NPU board power vs inference rate (flat to ~60 fps)
    sw = {int(k): v for k, v in data["npu_sweep"].items() if k.isdigit()}
    xs = sorted(sw)
    ax2.plot(xs, [sw[x] for x in xs], "-o", color="#1d9e75", lw=2.4, ms=5)
    ax2.axhline(idle, color="#ccc", ls=":", lw=1)
    ax2.text(xs[-1], idle - 0.05, f"idle {idle} W", ha="right", fontsize=8, color="#999")
    ax2.axvspan(0, 60, color="#1d9e75", alpha=0.06)
    ax2.text(30, sw[max(sw)] * 0.9, "flat to ~60 fps\n(+0.09 W)", fontsize=9,
             color="#1d7a5a", ha="center")
    ax2.set_xlabel("NPU inference rate (fps)", color=INK)
    ax2.set_ylabel("board power (W)", color=INK)
    ax2.set_title("NPU board power vs rate", color=INK, fontweight="medium", fontsize=11)

    for a in (ax, ax2):
        a.grid(axis="x" if a is ax else "y", color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("A733 whole-board power (RuiDeng TC66C, in-line, 30 s averages)",
                 color=INK, fontsize=12, fontweight="medium")
    fig.tight_layout()
    fig.savefig(OUT / "energy_npu_vs_cpu.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/energy_npu_vs_cpu.png")


if __name__ == "__main__":
    fidelity_chart()
    latency_chart()
    energy_chart()
