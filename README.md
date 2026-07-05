# MediaPipe FaceLandmarker on the VeriSilicon VIP9000 NPU (Allwinner A733)

**The first public port of Google's MediaPipe FaceLandmarker to a VeriSilicon
VIP9000 NPU** — face detection, 478-point face mesh, and 52 blendshapes,
compiled to run on the 3 TOPS NPU of the Allwinner A733 (Radxa Cubie A7A / A7Z
/ A7S, Orange Pi 4 Pro).

![status](https://img.shields.io/badge/conversion-validated-1d9e75)
![status](https://img.shields.io/badge/on--device-benchmarked-1d9e75)
![speed](https://img.shields.io/badge/NPU_int16-5.6x_CPU_(iso--scope)-1d9e75)
![energy](https://img.shields.io/badge/energy-~7x_less_than_CPU-1d9e75)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)
![platform](https://img.shields.io/badge/SoC-Allwinner_A733-444)
![npu](https://img.shields.io/badge/NPU-VeriSilicon_VIP9000-444)

---

## Why this exists

Nobody had published it. A [survey of the toolchain landscape](docs/RESEARCH.md)
(19 sources, adversarially fact-checked) confirmed that no public port of
MediaPipe's face models to a VIP9000-class NPU existed — Radxa's own model zoo
for the A7A ships ~20 models, **none of them MediaPipe**. There are open
requests for exactly this capability ([onnxruntime#28244](https://github.com/microsoft/onnxruntime/issues/28244),
[Frigate#23418](https://github.com/blakeblackshear/frigate/discussions/23418)).

If you have an A733 board and want face landmarks / mesh / blendshapes off the
CPU and onto the NPU, this repo is your starting point: the **conversion
recipe**, the **compiled NBG binaries**, the **fidelity validation**, and a
**benchmark harness**.

## Status

| Stage | State |
|---|---|
| Extract the 3 TFLite models from `face_landmarker.task` | ✅ done |
| Import into ACUITY (BlazeFace, Attention Mesh, Blendshapes) | ✅ 0 errors |
| Quantize (fp16 / bf16 / int16 / int8-pcq) | ✅ done |
| Export to NBG for the exact A733 NPU target | ✅ 0 errors |
| Numerically validate vs reference TFLite | ✅ FP16 lossless |
| Generate the OpenVX C runner projects | ✅ done |
| **Run on the physical NPU (`vpm_run`, VIPLite v2.0)** | ✅ **done** |
| **Latency benchmark, NPU vs CPU** | ✅ **done** |

Both halves are now complete: **the models convert losslessly, and they run on
the physical VIP9000** on a Radxa Cubie A7A (A733, kernel `5.15.147-21-a733`,
`/dev/vipcore`, ai-sdk VIPLite v2.0). The headline result overturns the naive
expectation — see below.

## Results

### Conversion fidelity (measured)

Every converted model was compared against the reference TFLite/XNNPACK output
on identical inputs. **FP16 and INT16 are both near-lossless**; naive INT8
wrecks the landmark and detector heads — exactly the trade-off this chart makes
visible:

![fidelity](charts/fidelity.png)

Headline numbers (mean abs error vs TFLite):

| model | output | FP16 | INT16 | INT8 (pcq) |
|---|---|---|---|---|
| face_detector | box logits | 0.25 | **0.25** | 1.97 ❌ |
| face_landmarks | 478 pts (px) | 0.09 px | **0.12 px** | 1.78 px ❌ |
| face_blendshapes | 52 scores | 0.0002 | **0.0001** | 0.0095 |

INT8 is rejected. FP16 and INT16 are both accurate enough — **but which one you
actually run should be decided by _speed on the real silicon_, not fidelity
alone. That is where the on-device benchmark below is decisive.**

### On-device latency: NPU vs CPU (measured, iso-scope)

![latency](charts/latency_npu_vs_cpu.png)

NPU measured with `vpm_run` (VIPLite v2.0, averaged over 100 loops) on the
physical VIP9000. CPU measured at the **same scope** — raw per-model
TFLite/XNNPACK fp32 inference (tflite-runtime, median of 100 loops, pinned to a
Cortex-A76 with `taskset`). Pure inference vs pure inference, no pre/post:

| model | NPU **int16** | CPU fp32 (1× A76) | CPU fp32 (2× A76) | NPU fp16 |
|---|---|---|---|---|
| face_detector | **0.67 ms** | 3.86 ms | 2.39 ms | 21.8 ms |
| face_landmarks | **3.16 ms** | 17.75 ms | 12.84 ms | 98.1 ms |
| face_blendshapes | **0.48 ms** | 2.63 ms | 1.71 ms | 14.6 ms |
| **3-model total** | **4.31 ms** | 24.2 ms | 16.9 ms | 134 ms |

**The counter-intuitive results — the whole point of running it on real
silicon:**

- **NPU int16 is 5.6× faster than one A76 core, 3.9× faster than both** — at
  matched output accuracy (int16 is 0.12 px off the fp32 reference). Against
  MediaPipe's full CPU frame (~35 ms end-to-end including pre/post) the gap is
  ~8×, a different scope stated separately on purpose. And the NPU offloads the
  CPU entirely — the thermal/power win that matters for always-on installs.
- **FP16 is a trap on this NPU, and the cycle counters prove why:** the fp16
  graph executes **~29× more cycles** than int16 (98.4M vs 3.4M for the
  landmarks model) at the same effective clock (~1 GHz). The NANO-DI's fast MAC
  arrays are **integer-only**; fp16 convolutions fall back to a slow
  programmable path. This is a hardware mapping limit — no export flag fixes it.
  On a bigger VIP9000 with a real FP16 pipe the story would differ, but on this
  chip **INT16 is the correct precision**, and it costs no meaningful accuracy.
- **I/O overhead is small:** wall time vs pure-compute time differ by ~0.25 ms
  on the largest model (256×256×3 input DMA + output read + dispatch), ~7% of
  compute. Network create/prepare (~6 ms) happens once at startup. A chained
  3-model runner should budget ~5-6 ms/frame NPU-side plus CPU pre/post
  (letterbox, anchor decode + NMS, crop, tensor packing).

Raw numbers, cycle counts and method: [benchmark/results/latency.json](benchmark/results/latency.json).

### Energy: the NPU's real win (measured at the wall)

Latency is only half the story. Power was measured on the whole board with an
in-line **RuiDeng TC66C** USB meter (30 s averages, CPU utilisation tracked to
confirm the watts are the NPU's, not hidden CPU work):

![energy](charts/energy_npu_vs_cpu.png)

- Running the same 3-model face pipeline at **15 fps**, the NPU adds **+0.11 W**
  over the 2.42 W idle, versus **+0.77 W** for a single Cortex-A76 — **~7× less
  power for the same result**, at matched output accuracy.
- The NPU's board power is **nearly flat from 5 to 60 fps** (2.50 → 2.59 W): you
  can raise the inference rate almost for free up to ~60 fps.
- During NPU inference the A76 sits at **1–3 % utilisation** (416–627 MHz), so
  the measured watts are genuinely the NPU's. A native chained C runner (no
  per-call `vpm_run` setup) would push them a little lower still.

Raw numbers: [benchmark/results/power.json](benchmark/results/power.json).

## How it works

```
camera frame
   │
   ▼  128×128            ▼  256×256 (face crop)        ▼  146 landmarks (px)
┌──────────────┐   ┌───────────────────────┐   ┌────────────────────┐
│ face_detector│──▶│ face_landmarks_detector│──▶│  face_blendshapes  │
│  (BlazeFace) │   │   (Attention Mesh)     │   │   (MLP-Mixer)      │
└──────────────┘   └───────────────────────┘   └────────────────────┘
   896 anchors          478 × 3D points              52 scores (eyeLook*, …)
```

All three are compiled to **NBG** (VeriSilicon network binary graph) and chained
through the VIPLite runtime. The non-obvious bits that took digging:

- The **`float16` quantizer** in ACUITY 6.30.22 (undocumented by Radxa) makes
  the conversion lossless — but **`int16` is the one you deploy**: on the
  NANO-DI it is ~30× faster for near-identical accuracy (see the benchmark).
- The **NPU target** for the A733 is `VIP9000NANODI_PID0X1000003B`.
- The **146-landmark subset** fed to the blendshapes model, extracted from
  MediaPipe's `face_blendshapes_graph.cc` ([convert/landmarks_subset.py](convert/landmarks_subset.py)).

Full recipe: [convert/README.md](convert/README.md).

## Repository layout

```
models/      the source MediaPipe models, vendored (face_landmarker.task + 3 TFLite)
compiled/    6 NBG binaries (fp16 + int16) + generated OpenVX C projects
convert/     reproducible ACUITY pipeline + numeric validation script
benchmark/   latency harness + RUNTIME.md (how to run the NBG on the NPU)
             results/  measured latency.json + fidelity.json
charts/      chart generator (run after a benchmark to refresh the PNGs)
docs/        RESEARCH.md — the state-of-the-art survey behind this work
```

**Self-contained by design.** The models are *vendored* in `models/` and the
compiled NBG in `compiled/`, so the repo keeps working even if the upstream
MediaPipe download, the ai-sdk mirror, or Radxa's model zoo disappear. The only
things you cannot legally vendor are the two proprietary VeriSilicon toolchains,
so they are pinned by exact version instead:
- **conversion**: Allwinner ACUITY docker `ubuntu-npu:v2.0.10.2` (target
  `VIP9000NANODI_PID0X1000003B`) — see [convert/README.md](convert/README.md).
- **runtime**: ai-sdk VIPLite **v2.0** (`libNBGlinker.so`, `libVIPhal.so`) + the
  `/dev/vipcore` kernel driver from the A733 BSP — see
  [benchmark/RUNTIME.md](benchmark/RUNTIME.md).

## Run the NBG on the NPU

The compiled binaries in `compiled/*/network_binary.nb` run today with the
`vpm_run` tool from ai-sdk. Full copy-paste recipe (driver check, library
install, build, benchmark loop) is in **[benchmark/RUNTIME.md](benchmark/RUNTIME.md)**.

## Reproduce the conversion

The models are already vendored in `models/`, so you can skip the download and
go straight to ACUITY. You need the Allwinner ACUITY docker image
(`ubuntu-npu:v2.0.10.2`) for the A733; see [convert/README.md](convert/README.md)
for the exact `pegasus` commands. In short:

```bash
# models/ is already populated; ./fetch_models.sh re-downloads them if ever needed
python convert/prepare_workspaces.py    # build calibration workspaces
# ... pegasus import → quantize (float16 | int16) → export ovxlib  (see convert/README.md)
python charts/make_charts.py            # refresh charts from benchmark/results/
```

## Hardware

Any **Allwinner A733** board (VeriSilicon VIP9000, 3 TOPS INT8, FP16/BF16/INT16
native): Radxa Cubie **A7A / A7Z / A7S**, Orange Pi 4 Pro. The compiled NBG are
target-specific to this NPU; other VIP9000 variants need a re-export with the
matching `--optimize` target.

## Roadmap

- [x] First on-device inference (`vpm_run` against VIPLite v2.0)
- [x] Latency benchmark, NPU vs CPU (iso-scope) — **int16 wins, 5.6× vs CPU**
- [x] FP16 root cause via cycle counters — integer-only MAC arrays, unfixable
- [ ] Full C runner: anchor decoding, NMS, crop, the 3-model chain
- [ ] Live demo (the [Watch Me](https://github.com/arnaudlvq/watchme) eye-contact clock that motivated this)
- [ ] Python path via `tflite-vx-delegate` (exploratory, see RESEARCH.md)

## Topics

`mediapipe` · `npu` · `verisilicon` · `vip9000` · `allwinner` · `a733` ·
`radxa` · `edge-ai` · `face-landmarks` · `face-mesh` · `blazeface` · `tflite` ·
`acuity` · `quantization` · `edge-inference` · `single-board-computer`

## License & attribution

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The compiled binaries
are derivatives of Google's MediaPipe FaceLandmarker models, distributed under
Apache-2.0. This is an independent project, not affiliated with Google,
VeriSilicon, Allwinner, or Radxa.
