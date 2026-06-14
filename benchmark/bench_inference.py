"""Inference benchmark harness for the Watch Me pipeline on the Radxa Cubie A7A.

Measures, for a given backend (cpu today, npu once ported):
  - per-frame latency distribution (p50 / p95 / p99 / max)
  - sustained fps
  - SoC temperatures (all /sys/class/thermal zones) and CPU frequencies
  - throttling events (frequency drops below nominal)

Two modes:
  peak     - short burst at maximum rate (no pacing), ~60 s
  longrun  - paced at the artwork's real rate for hours (default 2 h)

Usage (on the board):
  .venv/bin/python npu/bench/bench_inference.py --mode peak
  .venv/bin/python npu/bench/bench_inference.py --mode longrun --duration 7200 --fps 9
  .venv/bin/python npu/bench/bench_inference.py --source image --image test_face.png

Outputs a JSONL sample log and a JSON summary next to this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2

from eye_contact_detector import EyeContactDetector
from main import Settings

THERMAL_DIR = Path("/sys/class/thermal")
CPUFREQ_DIR = Path("/sys/devices/system/cpu")


def read_temperatures() -> dict[str, float]:
    temps = {}
    for zone in sorted(THERMAL_DIR.glob("thermal_zone*")):
        try:
            name = (zone / "type").read_text().strip()
            temps[name] = int((zone / "temp").read_text()) / 1000.0
        except (OSError, ValueError):
            continue
    return temps


def read_cpu_freqs_mhz() -> dict[str, int]:
    freqs = {}
    for policy in sorted(CPUFREQ_DIR.glob("cpufreq/policy*")):
        try:
            freqs[policy.name] = int((policy / "scaling_cur_freq").read_text()) // 1000
        except (OSError, ValueError):
            continue
    return freqs


class ImageCap:
    """Camera stand-in replaying a fixed image (reproducible benchmarks)."""

    def __init__(self, path: str) -> None:
        self.frame = cv2.imread(path)
        if self.frame is None:
            raise SystemExit(f"could not read {path}")

    def read(self):
        return True, self.frame.copy()

    def release(self) -> None:
        pass


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct / 100))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["peak", "longrun"], default="peak")
    parser.add_argument("--duration", type=float, default=None, help="seconds")
    parser.add_argument("--fps", type=float, default=9.0, help="pacing for longrun mode")
    parser.add_argument("--backend", choices=["cpu"], default="cpu")  # npu: coming
    parser.add_argument("--source", choices=["camera", "image"], default="camera")
    parser.add_argument("--image", default=str(Path(__file__).parent / "test_face.png"))
    parser.add_argument("--out", default=None, help="output basename")
    args = parser.parse_args()

    duration = args.duration or (60.0 if args.mode == "peak" else 7200.0)
    paced_interval = 0.0 if args.mode == "peak" else 1.0 / args.fps
    out_base = args.out or f"bench_{args.backend}_{args.mode}_{int(time.time())}"
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    detector = EyeContactDetector(Settings())
    if args.source == "image":
        detector.cap.release()
        detector.cap = ImageCap(args.image)

    latencies: list[float] = []
    samples_path = out_dir / f"{out_base}.jsonl"
    t_start = time.monotonic()
    last_sample = 0.0
    frames = 0

    print(f"backend={args.backend} mode={args.mode} duration={duration:.0f}s -> {samples_path}")
    with samples_path.open("w") as log:
        while (elapsed := time.monotonic() - t_start) < duration:
            t0 = time.monotonic()
            detector.detect_eye_contact()
            latency_ms = (
                detector.last_reading.latency_ms
                if detector.last_reading is not None
                else (time.monotonic() - t0) * 1000
            )
            latencies.append(latency_ms)
            frames += 1

            # One environment sample (temps, freqs) every 10 s
            if elapsed - last_sample >= 10.0:
                last_sample = elapsed
                sample = {
                    "t_s": round(elapsed, 1),
                    "latency_ms": round(latency_ms, 2),
                    "fps_so_far": round(frames / elapsed, 2) if elapsed else None,
                    "temps_c": read_temperatures(),
                    "cpu_mhz": read_cpu_freqs_mhz(),
                }
                log.write(json.dumps(sample) + "\n")
                log.flush()
                temp_max = max(sample["temps_c"].values()) if sample["temps_c"] else float("nan")
                print(
                    f"  {elapsed:7.0f}s  lat {latency_ms:6.1f} ms  "
                    f"fps {sample['fps_so_far']:6.2f}  Tmax {temp_max:.1f}C"
                )

            spare = paced_interval - (time.monotonic() - t0)
            if spare > 0:
                time.sleep(spare)

    detector.release()

    summary = {
        "backend": args.backend,
        "mode": args.mode,
        "duration_s": round(time.monotonic() - t_start, 1),
        "frames": frames,
        "fps_mean": round(frames / (time.monotonic() - t_start), 2),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "p99": round(percentile(latencies, 99), 2),
            "max": round(max(latencies), 2),
        },
        "final_temps_c": read_temperatures(),
    }
    summary_path = out_dir / f"{out_base}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
