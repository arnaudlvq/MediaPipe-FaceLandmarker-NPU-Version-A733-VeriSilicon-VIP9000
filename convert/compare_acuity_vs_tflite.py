"""Numeric validation, run INSIDE the ubuntu-npu container.

For each model: run the reference TFLite interpreter and the ACUITY-imported
graph (pegasus inference) on the same input, in float and quantized modes,
and report per-output max/mean absolute differences.

Usage: python3 /workspace/compare_acuity_vs_tflite.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf

PEGASUS = "/root/acuity-toolkit-whl-6.30.22/bin/pegasus.py"
WORK = Path("/workspace")

MODELS = {
    "face_detector": {"prep": "detector"},
    "face_landmarks_detector": {"prep": "landmarks"},
    "face_blendshapes": {"prep": "tensor"},
}


def load_input(name: str, spec: dict) -> np.ndarray:
    work = WORK / name
    if spec["prep"] == "tensor":
        return np.load(work / "calib" / "landmarks_000.npy").astype(np.float32)
    import cv2

    img = cv2.imread(str(work / "calib" / "face_000.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if spec["prep"] == "detector":  # 128x128, [-1, 1]
        img = (img - 127.5) * 0.0078431
    else:  # landmarks: 256x256, [0, 1]
        img = img * 0.0039216
    return img[None]


def run_tflite(name: str, x: np.ndarray) -> list[np.ndarray]:
    interp = tf.lite.Interpreter(model_path=str(WORK / name / f"{name}.tflite"))
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]


def run_pegasus(name: str, dtype: str) -> list[np.ndarray]:
    work = WORK / name
    out_dir = work / f"inf_{dtype}"
    cmd = [
        "python3", PEGASUS, "inference",
        "--model", f"{name}.json",
        "--model-data", f"{name}.data",
        "--with-input-meta", f"{name}_inputmeta.yml",
        "--dtype", dtype,
        "--device", "CPU",
        "--output-dir", str(out_dir),
    ]
    if dtype == "quantized":
        cmd += ["--model-quantize", f"{name}.quantize"]
    subprocess.run(cmd, cwd=work, capture_output=True, check=True)
    tensors = []
    for f in sorted(out_dir.glob("*.tensor")) or sorted(out_dir.rglob("*out*.txt")):
        tensors.append(np.loadtxt(f, dtype=np.float32).ravel())
    return tensors


def main() -> None:
    report = {}
    for name, spec in MODELS.items():
        x = load_input(name, spec)
        ref = [t.ravel() for t in run_tflite(name, x)]
        entry = {}
        for dtype in ("float32", "quantized"):
            try:
                got = run_pegasus(name, dtype)
            except subprocess.CalledProcessError as e:
                entry[dtype] = f"pegasus failed: {e.stderr.decode()[-200:]}"
                continue
            # Match outputs by size (ACUITY may reorder)
            diffs = []
            for r in ref:
                match = min(
                    (g for g in got if g.size == r.size),
                    key=lambda g: float(np.abs(g - r).mean()),
                    default=None,
                )
                if match is None:
                    diffs.append({"size": int(r.size), "error": "no size-matched output"})
                    continue
                diffs.append(
                    {
                        "size": int(r.size),
                        "max_abs": round(float(np.abs(match - r).max()), 5),
                        "mean_abs": round(float(np.abs(match - r).mean()), 6),
                        "ref_range": [round(float(r.min()), 3), round(float(r.max()), 3)],
                    }
                )
            entry[dtype] = diffs
        report[name] = entry
        print(f"=== {name}")
        print(json.dumps(entry, indent=1))

    (WORK / "validation_report.json").write_text(json.dumps(report, indent=1))
    print("\nreport -> /workspace/validation_report.json")


if __name__ == "__main__":
    main()
