# Running the compiled NBG on the physical VIP9000

This is the exact, reproduced-on-hardware recipe that produced
[`results/latency.json`](results/latency.json). It runs the compiled
`compiled/*/network_binary.nb` graphs on the NPU of a **Radxa Cubie A7A**
(Allwinner A733), kernel `5.15.147-21-a733`.

Everything here is pinned so the repo stays reproducible even if the upstream
sources move. The two pieces that cannot be vendored (proprietary VeriSilicon
binaries) are named by exact version.

---

## 0. What you need on the board

| Piece | What / where | Vendored here? |
|---|---|---|
| NBG graphs | `compiled/<model>_nbg_{int16,fp16}/network_binary.nb` | ✅ yes |
| NPU kernel driver | `/dev/vipcore` (from the A733 BSP kernel) | ❌ ships in the board image |
| VIPLite runtime v2.0 | `libNBGlinker.so`, `libVIPhal.so` | ❌ from ai-sdk (pinned below) |
| `vpm_run` tool | ai-sdk `examples/vpm_run` | ❌ built from source below |

The NPU driver node `/dev/vipcore` must exist:

```bash
ls -l /dev/vipcore          # character device -> NPU is exposed
dmesg | grep -iE 'vip|npu|galcore' | head
```

If it is missing, your kernel was not built with the VIP driver — you need the
A733 BSP image (Radxa's official Debian/Ubuntu build has it).

## 1. Get the VIPLite runtime + vpm_run (ai-sdk)

The runtime libraries and the `vpm_run` harness come from ZIFENG278's ai-sdk,
which packages Allwinner's `viplite-tina` for the A733. **Pinned: VIPLite
`v2.0`** — the library path below is the contract; if this repo ever needs it
and ai-sdk is gone, the two files you must obtain are `libNBGlinker.so` and
`libVIPhal.so` from a v2.0 `viplite-tina` for aarch64.

```bash
cd ~
git clone --depth 1 https://github.com/ZIFENG278/ai-sdk.git

# confirm the A733 NPU version the SDK expects
grep -iE 'NPU_VERSION|NPU_SW_VERSION' ~/ai-sdk/machinfo/a733/config.mk

# v2.0 libraries for aarch64
ls ~/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0/
```

Install the two runtime libs so the loader finds them:

```bash
sudo cp ~/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0/libNBGlinker.so /usr/lib/
sudo cp ~/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0/libVIPhal.so   /usr/lib/
sudo ldconfig
ldconfig -p | grep -iE 'NBGlinker|VIPhal'      # should list both
```

## 2. Build vpm_run

```bash
sudo apt-get install -y build-essential           # needs a C toolchain + libc headers
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
cd ~/ai-sdk/examples/vpm_run
make AI_SDK_PLATFORM=a733 CROSS_COMPILE=          # CROSS_COMPILE empty = build natively on the board
ls -l ./vpm_run                                   # the binary
```

## 3. Benchmark a graph

`vpm_run` takes a tiny sample file that points at the NBG and its input
tensor(s). Latency is input-independent for a fixed graph, so a zeroed dummy
input of the right byte size is enough to measure compute time.

Input byte sizes (2 bytes/element — these graphs take a **16-bit** input tensor):

| model | input tensor | bytes |
|---|---|---|
| face_detector | 128×128×3 | 98304 |
| face_landmarks_detector | 256×256×3 | 393216 |
| face_blendshapes | 146×2 | 584 |

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
VPM=~/ai-sdk/examples/vpm_run/vpm_run
NB=~/path/to/compiled/face_landmarks_detector_nbg_int16/network_binary.nb

mkdir -p /tmp/bench && cd /tmp/bench
cp "$NB" ./network_binary.nb
head -c 393216 /dev/zero > input_0.dat
printf '[network]\n./network_binary.nb\n[input]\n./input_0.dat\n' > sample.txt

# -l 100 = average over 100 loops
"$VPM" -s sample.txt -l 100 2>&1 | grep -iE 'profile|inference|run time'
```

Read the reported **profile inference time** (ms). Loop this over all three
models × `{int16, fp16}` to reproduce [`results/latency.json`](results/latency.json).
The one-shot driver script that produced our numbers is reproduced at the bottom
of this file.

## 4. Results we measured

On the Radxa Cubie A7A (A733), VIPLite v2.0, averaged over 100 loops:

| model | NPU **int16** | NPU fp16 |
|---|---|---|
| face_detector | 0.67 ms | 21.75 ms |
| face_landmarks_detector | 3.16 ms | 98.10 ms |
| face_blendshapes | 0.48 ms | 14.60 ms |
| **total** | **4.31 ms** | 134.45 ms |

CPU reference at the same scope (raw per-model TFLite/XNNPACK fp32, pinned to
one Cortex-A76): **24.2 ms** total (17.75 ms for the landmarks model alone).
MediaPipe's full CPU frame including pre/post: ≈ 35 ms.

**INT16 is the deploy target** — ~30× faster than FP16 on the NANO-DI (cycle
counters show fp16 executes ~29× more cycles: the MAC arrays are integer-only),
5.6× faster than the CPU at iso-scope, and near-lossless (0.12 px landmark
error). I/O overhead is ~0.25 ms per inference (wall vs compute time); network
create/prepare (~6 ms) is one-time. See the repo README for the full discussion.

---

## Appendix — the driver script

```bash
set +e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
VPM=$HOME/ai-sdk/examples/vpm_run/vpm_run
COMP=./compiled                          # this repo's compiled NBGs
sz_face_detector=98304
sz_face_landmarks_detector=393216
sz_face_blendshapes=584
for prec in int16 fp16; do
  for m in face_detector face_landmarks_detector face_blendshapes; do
    eval SZ=\$sz_$m
    d=/tmp/npu_bench/${m}_${prec}; mkdir -p "$d"; cd "$d"
    cp "$COMP/${m}_nbg_${prec}/network_binary.nb" ./network_binary.nb
    head -c "$SZ" /dev/zero > input_0.dat
    printf '[network]\n./network_binary.nb\n[input]\n./input_0.dat\n' > sample.txt
    echo "===== $m  $prec ====="
    "$VPM" -s sample.txt -l 100 2>&1 | grep -iE 'profile|inference|run time' | head -5
  done
done
```
