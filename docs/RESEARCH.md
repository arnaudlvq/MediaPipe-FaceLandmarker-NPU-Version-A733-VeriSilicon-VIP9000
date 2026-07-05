# Porting MediaPipe FaceLandmarker to the Allwinner A733 VIP9000 NPU, state of the art

*Deep research, 2026-06-12. 5 axes, 19 primary sources, 25 claims verified by
adversarial triple-vote (24 confirmed, 1 refuted).*

## Overall verdict

**We found no public port.** No port of BlazeFace, Face Mesh / Attention Mesh or
GHUM Blendshapes to a VeriSilicon VIP9000 NPU appears to exist, neither at Radxa
(the [official A7A Model Zoo](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/model-zoo)
lists ~20 models: YOLO v3→v26, RetinaFace, MobileNet, ResNet50… zero MediaPipe
models), nor in the Khadas (A311D), NXP (i.MX 8M Plus) or Mesa/Teflon
communities. The closest prior art is RetinaFace (face detection only). The
"MediaPipe" page of the Radxa A7A docs is a plain CPU `pip install mediapipe`.

→ We did not find any public prior port, which is why a clean, well-documented
repo seems worth publishing.

## The four paths

| Path | Verdict | Detail |
|---|---|---|
| **(a) ACUITY → NBG + VIPLite (C)** | ✅ **Mature, documented, recommended** | The only chain documented end to end by Radxa for the Cubie A7A. Accepts TFLite as input (no need to go through ONNX). `pegasus_import/quantize/inference/export_ovx` scripts. Generates an **OpenVX C project**, aligned with the C-rewrite goal. No documented on-device Python path; the Model Zoo examples are C++ against VIPLite 2.0.3.2-AW. |
| **(b) TIM-VX + tflite-vx-delegate** | ⚠️ **Real but unproven on A733** | Official VeriSilicon TFLite delegate, usable from Python (`load_delegate`), demonstrated on Khadas VIM3/A311D, in production at NXP. Supports VIPLite through the "no-kernel" SDK (≥6.4.22). BUT: the A733 is not a reference board, the build requires the Allwinner BSP userspace libs (strictly matched versions), and open issues report wrong outputs/segfaults on some models. |
| **(c) Mesa Teflon / etnaviv** | ❌ **Dead for the A733** | Fully open-source stack but limited to VIPNano-QI (A311D), VIPNano-SI+ (i.MX 8M Plus) and RK3588; UINT8 CNN only; the A733's VIP9000 is not covered, and Tomeu Vizoso pivoted to Rockchip in 2025. Worth watching long term. |
| **(d) ONNX Runtime / others** | ❌ **Nonexistent** | No VIPLite execution provider, which is the subject of [open issue #28244](https://github.com/microsoft/onnxruntime/issues/28244) asking for exactly A733/T527 support. |

## The #1 technical risk: operator coverage

The CNN ops of **BlazeFace** and the Face Mesh trunk pass everywhere (Conv2d,
DepthwiseConv2d, Prelu, Reshape, Pad, FullyConnected: all "yes" in
[op_status.md](https://github.com/VeriSilicon/tflite-vx-delegate/blob/main/op_status.md)).
However **GELU and LayerNorm are missing** from the delegate's operator matrix
(checked in the code: zero occurrences in op_map.cc), and those are the building
blocks of the attention mechanisms in **Attention Mesh** and the **GHUM
Blendshapes** (MLP-Mixer style). The conversion feasibility of these two models
is settled nowhere: **only a real `pegasus_import` attempt will tell**.

Fallback plans if it blocks: (1) port only BlazeFace + landmarks and keep the
blendshapes model on CPU (it takes *landmarks* as input, not the image, and is
tiny); (2) substitute Face Mesh V1 (no attention) for V2.

## Quantization: the strategy

ACUITY modes documented on A7A: `uint8`, `pcq` (per-channel int8), `int16`,
`bf16`, `float` (no quantization), + hybrid quantization. **FP16 is not an
explicit target** (the MediaPipe models ship as float16), the 16-bit options are
BF16/INT16, or the `float` mode whose internal execution is unspecified.

Pitfall documented by Radxa itself: in their MobileNetV2 example, **plain uint8
degrades accuracy** (wrong predicted class), while `pcq` and `int16` stay
faithful to float. For *landmark regression* models (more sensitive than a
classifier), start directly on **pcq or int16**, with the
[precision optimization page](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-quant-acc-improve)
(mixed quantization) as a fallback.

## Performance: no published data, we broke ground

No latency figure exists for BlazeFace/Face Mesh style models on the A733's 3
TOPS VIP9000. The only references in the family: A311D 5 TOPS, MobileNetV1 ≈
5.5-6.6 ms. Assumed open question: for models this small (1-5 MFLOPs), does the
NPU gain beat the memory-transfer cost versus the A76 + XNNPACK? **That is
exactly what our bench (`npu/bench/`) measured**, p50/p95/p99 latency, fps,
temperatures, at peak and over endurance, CPU vs NPU on the same grid.
Complementary on-device tools: `vpm_run` and `NBinfo` (documented by Radxa, no
published result found).

## Honesty caveats

- Converting Attention Mesh and the GHUM Blendshapes was **unproven** going in,
  the main risk, cleared first (a conversion attempt before writing any C code).
- ACUITY's `float` mode may run as native FP16 on the NPU, speed unknown.
- The A311D figures do not transpose directly (different NPU generation).
- "Documented" ≠ "reliable": open issues on the delegate (wrong INT8 outputs,
  segfaults on some models).
- Radxa docs subject to link rot (one URL changed during the research itself).

## Main sources

[Radxa NPU dev (A7A)](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev) ·
[ACUITY usage](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-acuity-usage) ·
[Model Zoo A7A](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/model-zoo) ·
[tflite-vx-delegate](https://github.com/VeriSilicon/tflite-vx-delegate) ·
[TIM-VX](https://github.com/VeriSilicon/TIM-VX) ·
[Khadas VIM3 vx-tflite](https://docs.khadas.com/products/sbc/vim3/npu/vx-tflite) ·
[Mesa Teflon](https://docs.mesa3d.org/teflon.html) ·
[Tomeu Vizoso blog](https://blog.tomeuvizoso.net/) ·
[onnxruntime #28244](https://github.com/microsoft/onnxruntime/issues/28244) ·
[Frigate #23418](https://github.com/blakeblackshear/frigate/discussions/23418) ·
[acuitylite (VeriSilicon)](https://verisilicon.github.io/acuitylite/README.html)
