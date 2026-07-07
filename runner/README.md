# Chained NPU runner

One call per frame: RGB image in, 478 face landmarks + 52 blendshapes out.
The three FaceLandmarker models run on the VIP9000 NPU; this runner does the
CPU glue between them (letterbox, anchor decoding, face crop, landmark subset)
that `vpm_run` does not do.

Status: validated on the board (Radxa Cubie A7A). Against the official
MediaPipe CPU output on the same image, the chain agrees to **0.58 px mean
(p95 0.94 px, max 1.54 px)** on the 478 landmarks and 0.012 mean absolute
delta on the 52 blendshapes. End-to-end: **9.6 ms/frame** (p50, pinned to one
A76; p99 11.3 ms), of which 4.3 ms is NPU compute and ~5 ms is CPU glue
(fixed-point bilinear resampling fused with the int16 quantization, written
straight into the mapped NPU buffers). MediaPipe's full CPU frame on the same
board is 32.4 ms, so the chain is 3.4x faster end to end and leaves the big
cores mostly idle.

## Use it

On the board (Radxa Cubie A7A, NPU runtime installed per
[benchmark/RUNTIME.md](../benchmark/RUNTIME.md), steps 0 and 1):

```bash
cd runner
make                                   # builds fl_run + libfacelandmarker_npu.so
ffmpeg -i your_photo.jpg -pix_fmt rgb24 face.ppm
./fl_run --models ../compiled --ppm face.ppm --loop 100
```

Output: a JSON line with the 478 landmarks (frame pixels), the 52 blendshape
scores, per-stage milliseconds, and a latency summary (p50/p90/p99) on stderr.

From Python (or any language with a C FFI), use the shared library:

```python
import ctypes

lib = ctypes.CDLL("./libfacelandmarker_npu.so")

class Result(ctypes.Structure):
    _fields_ = [("face_present", ctypes.c_int),
                ("detect_score", ctypes.c_float),
                ("presence_score", ctypes.c_float),
                ("landmarks", ctypes.c_float * (478 * 3)),
                ("blendshapes", ctypes.c_float * 52),
                ("roi", ctypes.c_float * 4),
                ("ms", ctypes.c_double * 4)]

lib.fl_create.restype = ctypes.c_void_p
ctx = lib.fl_create(b"../compiled")
res = Result()
lib.fl_process_rgb(ctx, frame_rgb.ctypes.data, w, h, ctypes.byref(res))
```

The full C API is three functions: `fl_create(models_dir)`,
`fl_process_rgb(ctx, rgb, w, h, &result)`, `fl_destroy(ctx)`. See
[src/facelandmarker.h](src/facelandmarker.h).

## How it works

```
frame (RGB888)                              measured (512x512 input, 1x A76)
  letterbox 128x128, [-1,1]          CPU    \
  face_detector                      NPU     } stage 1: 1.7 ms
  decode + pick best face            CPU    /
  rotated crop 256x256, [0,1]        CPU    \
  face_landmarks_detector            NPU     } stage 2: 6.6 ms
  map landmarks back to frame px     CPU    /
  subset 146 points, pixel coords    CPU    \
  face_blendshapes                   NPU     } stage 3: 0.9 ms
```

NPU compute is 4.31 ms of the 9.6 ms total; the CPU glue (~5 ms) is
fixed-point bilinear resampling fused with the int16 quantization, writing
straight into the mapped NPU input buffers (no intermediate float pass, no
extra copy). The quantization is exact at the range ends (255*257 = 65535 for
the [-1,1] input). Only the glue and the I/O dequantization run on the CPU,
the networks never touch it.

Each CPU stage replicates a MediaPipe calculator from the `face_landmarker.task`
graph, with the same constants:

| stage | MediaPipe calculator | constants |
|---|---|---|
| anchors | SsdAnchorsCalculator | input 128, strides 8,16,16,16, fixed size, 896 anchors |
| decode | TensorsToDetectionsCalculator | x/y/w/h scale 128, sigmoid clip 100, score min 0.5 |
| ROI | DetectionsToRects + RectTransformation | rotation from eye keypoints, scale 1.5, square |
| blendshapes input | LandmarksToTensorCalculator | 146-point subset, pixel coordinates |

The 146 indices are verbatim `kLandmarksSubsetIdxs` from MediaPipe
`face_blendshapes_graph.cc`. The blendshapes model standardizes its input
internally (its first ops are Mean/SquaredDifference/Rsqrt), which is why
plain pixel coordinates are the expected input.

Quantization at the NPU boundary (from the NBG metadata, dynamic fixed point):

| tensor | dtype | fixed point | conversion |
|---|---|---|---|
| detector input 128x128x3 | int16 | fl=15 | [-1,1] x 2^15 |
| detector regressors 896x16 | int16 | fl=7 | /128 |
| detector scores 896 | int16 | fl=8 | /256 |
| landmarks input 256x256x3 | int16 | fl=14 | [0,1] x 2^14 |
| landmarks output 1434 | int16 | fl=7 | /128 (crop pixels) |
| blendshapes input 146x2 | fp16 | none | IEEE half |
| blendshapes output 52 | int16 | fl=15 | /2^15 |

Two deliberate simplifications, fine for a single-viewer installation and
documented so you can lift them:

1. Top-1 detection instead of weighted non-max suppression. For multiple
   faces, port `WeightedNonMaxSuppression` into `decode_best()`.
2. The detector runs every frame instead of MediaPipe's skip-when-tracking.
   It costs 0.67 ms on the NPU; the state machine it saves is not worth it
   at this budget.

## Rebuild recipe

Requirements on the board: `build-essential` and the two VIPLite v2.0
libraries (`libNBGlinker.so`, `libVIPhal.so`) from the ai-sdk, installed as in
[benchmark/RUNTIME.md](../benchmark/RUNTIME.md). The Makefile defaults to
`~/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0`; override with
`make VIPLITE=/path/to/v2.0`.

Source layout, three files, no dependencies beyond libc and VIPLite:

- [src/vipnet.c](src/vipnet.c): a ~200 line VIPLite wrapper. Open an NBG,
  query its tensors (dims, dtype, fixed-point position), create and attach
  buffers, run, dequantize. This is the part you would reuse for any other
  model on this NPU.
- [src/facelandmarker.c](src/facelandmarker.c): the pipeline above.
- [src/main.c](src/main.c): the CLI (PPM in, JSON out, latency percentiles).

The call sequence in vipnet.c follows the vendor examples (`vpm_run`,
`libawnn_viplite` in the ai-sdk): create network from file, query I/O
properties, `vip_create_buffer`, `vip_prepare_network`, `vip_set_input/output`,
then per inference map + memcpy + cache flush, `vip_run_network`, cache
invalidate + map + dequantize.

To adapt to other models: change the model paths in `fl_create()` and the
pre/post stages; `vipnet.c` needs no changes, it reads every tensor property
from the NBG itself.
