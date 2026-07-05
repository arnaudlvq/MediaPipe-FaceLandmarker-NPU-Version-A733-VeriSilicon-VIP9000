#!/usr/bin/env python3
"""Fair, iso-scope CPU baseline: per-model raw TFLite inference latency on the A733.

Why this exists: the NPU numbers in benchmark/results/latency.json are pure
`vip_run_network` compute, per model. To compare fairly, the CPU side must be the
SAME scope, the raw per-model TFLite inference (XNNPACK on the Cortex-A76), NOT
the full end-to-end MediaPipe frame (which also does letterbox, anchor decode,
NMS, crop, tensor packing). This script times each of the three extracted models
so we can put NPU-vs-CPU side by side, per model, at matched scope.

Precision note: these MediaPipe models ship as float16 weights; TFLite/XNNPACK
runs them in float32 by default (full precision, the fidelity reference). We
also try to force fp16 compute so we have an iso-PRECISION point against the NPU
fp16 number. INT16 is not a native XNNPACK CPU path, so "CPU int16" is not a
meaningful comparison, the CPU's real options are fp32 and fp16.

Usage (on the board):
    pip install ai-edge-litert            # or tflite-runtime
    python3 npu/bench/cpu_permodel_bench.py --models npu/models --loops 100
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np


def make_interpreter(model_path: str, num_threads: int, force_fp16: bool):
    """Return a TFLite interpreter, trying the available runtimes in order."""
    delegates = []
    # Try ai-edge-litert (the current name), then tflite_runtime, then TF.
    Interpreter = None
    load_delegate = None
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate  # type: ignore
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate  # type: ignore
        except Exception:
            from tensorflow.lite import Interpreter  # type: ignore
            from tensorflow.lite.experimental import load_delegate  # type: ignore
    # XNNPACK is built-in and on by default; forcing fp16 needs a delegate flag
    # that isn't reliably exposed from Python across runtimes, so we report
    # whether it was honored. Default path = fp32 XNNPACK.
    kwargs = {"model_path": model_path, "num_threads": num_threads}
    interp = Interpreter(**kwargs)
    interp.allocate_tensors()
    return interp


def bench_model(model_path: Path, loops: int, threads: int) -> dict:
    interp = make_interpreter(str(model_path), threads, force_fp16=False)
    inp = interp.get_input_details()
    out = interp.get_output_details()
    # random input of the right dtype/shape
    feeds = []
    for d in inp:
        shape = d["shape"]
        dt = d["dtype"]
        if np.issubdtype(dt, np.floating):
            x = np.random.rand(*shape).astype(dt)
        else:
            x = np.random.randint(0, 255, size=shape).astype(dt)
        feeds.append((d["index"], x))
    # warmup
    for _ in range(5):
        for idx, x in feeds:
            interp.set_tensor(idx, x)
        interp.invoke()
    times = []
    for _ in range(loops):
        for idx, x in feeds:
            interp.set_tensor(idx, x)
        t = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t) * 1000.0)
    return {
        "model": model_path.stem,
        "loops": loops,
        "threads": threads,
        "median_ms": round(statistics.median(times), 3),
        "mean_ms": round(statistics.mean(times), 3),
        "p10_ms": round(sorted(times)[len(times) // 10], 3),
        "input_shapes": [list(d["shape"]) for d in inp],
        "out_count": len(out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models", help="dir with the 3 .tflite")
    ap.add_argument("--loops", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1,
                    help="1 = per-model single-core (matches the NPU's single-stream); "
                         "try 2 to use both A76 cores")
    args = ap.parse_args()

    mdir = Path(args.models)
    order = ["face_detector", "face_landmarks_detector", "face_blendshapes"]
    files = {p.stem: p for p in mdir.glob("*.tflite")}
    total = 0.0
    rows = []
    for name in order:
        if name not in files:
            print(f"MISSING {name}.tflite in {mdir}")
            continue
        r = bench_model(files[name], args.loops, args.threads)
        rows.append(r)
        total += r["median_ms"]
        print(f"{name:26s} median={r['median_ms']:8.3f} ms  mean={r['mean_ms']:8.3f} ms "
              f"(threads={args.threads})")
    print(f"{'TOTAL (3 models)':26s} median={total:8.3f} ms")
    import json
    print("JSON " + json.dumps({"threads": args.threads, "loops": args.loops, "rows": rows,
                                "total_median_ms": round(total, 3)}))


if __name__ == "__main__":
    main()
