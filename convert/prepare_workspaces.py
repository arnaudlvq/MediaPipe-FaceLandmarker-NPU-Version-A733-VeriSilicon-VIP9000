"""Build the pegasus/ACUITY conversion workspaces for the 3 face_landmarker models.

For each model, creates npu/convert/work/<name>/ with the .tflite, a
calibration set, dataset.txt and inputs_outputs.txt, ready to be mounted into
the ubuntu-npu docker container.

Calibration inputs:
  - face_detector: square face crops resized to 128x128 (RGB jpg)
  - face_landmarks_detector: tighter face crops resized to 256x256 (RGB jpg)
  - face_blendshapes: landmark tensors (1,146,2) as .npy - placeholder for the
    import test; the exact 146-subset extraction is a TODO for quantization.

Usage:
  .venv/bin/python npu/convert/prepare_workspaces.py [--images DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "npu" / "models"
WORK_DIR = Path(__file__).resolve().parent / "work"

sys.path.insert(0, str(ROOT))

IO_SPECS = {
    "face_detector": {"input": "input", "outputs": ["regressors", "classificators"]},
    "face_landmarks_detector": {
        "input": "input_12",
        "outputs": ["Identity", "Identity_1", "Identity_2"],
    },
    "face_blendshapes": {
        "input": "serving_default_input_points:0",
        "outputs": ["StatefulPartitionedCall:0"],
    },
}


_LANDMARKER = None


def _get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(ROOT / "face_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER


def detect_landmarks(image: np.ndarray) -> np.ndarray | None:
    """478 landmarks in pixel coordinates, or None if no face."""
    import mediapipe as mp

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    h, w = image.shape[:2]
    return np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]], np.float32)


def detect_face_box(image: np.ndarray) -> tuple[int, int, int, int]:
    """Rough face bounding square from the landmarks."""
    h, w = image.shape[:2]
    pts = detect_landmarks(image)
    if pts is None:
        side = min(h, w)
        return (w - side) // 2, (h - side) // 2, side, side
    cx, cy = (pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2
    side = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 1.6
    x = int(max(0, cx - side / 2))
    y = int(max(0, cy - side / 2))
    side = int(min(side, w - x, h - y))
    return x, y, side, side


def augment(image: np.ndarray) -> list[np.ndarray]:
    """Cheap augmentations so a small photo set still covers activation
    ranges: flips, brightness, scale. Real diverse captures remain better."""
    variants = [image, cv2.flip(image, 1)]
    for alpha, beta in ((0.7, -15), (1.3, 20)):
        variants.append(cv2.convertScaleAbs(image, alpha=alpha, beta=beta))
    h, w = image.shape[:2]
    zoomed = image[h // 8 : h - h // 8, w // 8 : w - w // 8]
    variants.append(cv2.resize(zoomed, (w, h)))
    return variants


def build_vision_workspace(name: str, size: int, images: list[Path]) -> None:
    work = WORK_DIR / name
    calib = work / "calib"
    calib.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODELS_DIR / f"{name}.tflite", work / f"{name}.tflite")

    entries = []
    i = 0
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        for variant in augment(image):
            x, y, w, h = detect_face_box(variant)
            crop = cv2.resize(variant[y : y + h, x : x + w], (size, size))
            out = calib / f"face_{i:03d}.jpg"
            cv2.imwrite(str(out), crop)
            entries.append(f"./calib/{out.name}")
            i += 1

    (work / "dataset.txt").write_text("\n".join(entries) + "\n")
    spec = IO_SPECS[name]
    (work / "inputs_outputs.txt").write_text(
        f"--inputs {spec['input']} --outputs {' '.join(spec['outputs'])}\n"
    )
    print(f"  {name}: {len(entries)} calibration crops ({size}x{size})")


def build_blendshapes_workspace(images: list[Path]) -> None:
    """Real calibration: the model input is the (x, y) pixel coordinates of
    the 146-landmark subset (see landmarks_subset.py), as MediaPipe's
    LandmarksToTensorCalculator produces them."""
    from landmarks_subset import BLENDSHAPES_LANDMARK_SUBSET

    name = "face_blendshapes"
    work = WORK_DIR / name
    calib = work / "calib"
    calib.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODELS_DIR / f"{name}.tflite", work / f"{name}.tflite")

    entries = []
    i = 0
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        for variant in augment(image):
            pts = detect_landmarks(variant)
            if pts is None:
                continue
            sample = pts[BLENDSHAPES_LANDMARK_SUBSET][None]  # (1, 146, 2) px
            out = calib / f"landmarks_{i:03d}.npy"
            np.save(out, sample.astype(np.float32))
            entries.append(f"./calib/{out.name}")
            i += 1

    (work / "dataset.txt").write_text("\n".join(entries) + "\n")
    spec = IO_SPECS[name]
    (work / "inputs_outputs.txt").write_text(
        f"--inputs '{spec['input']}' --outputs '{spec['outputs'][0]}'\n"
    )
    print(f"  {name}: {len(entries)} real landmark tensors (146 pts, pixel coords)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        default=str(ROOT / "npu" / "bench"),
        help="directory of face photos for calibration crops",
    )
    args = parser.parse_args()

    images = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not images:
        raise SystemExit(f"no images found in {args.images}")
    print(f"Building workspaces from {len(images)} image(s) -> {WORK_DIR}")

    build_vision_workspace("face_detector", 128, images)
    build_vision_workspace("face_landmarks_detector", 256, images)
    build_blendshapes_workspace(images)
    print("Done. Mount npu/convert/work/ into the ubuntu-npu container (see README).")


if __name__ == "__main__":
    main()
