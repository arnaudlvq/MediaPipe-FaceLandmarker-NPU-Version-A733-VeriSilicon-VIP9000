# Benchmark results

Measured on a Radxa Cubie A7A (Allwinner A733), committed as-is.

- `fidelity.json` — conversion accuracy vs the reference TFLite (per precision).
- `latency.json` — on-device NPU vs CPU per-model latency, with the fp16 cycle counts.
- `power.json` — whole-board power (RuiDeng TC66C, in-line) per config.

Regenerate the charts after editing these: `python charts/make_charts.py`.
