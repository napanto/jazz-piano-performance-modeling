# Training hardware

Hardware on which each reported model was trained, for the report's Methods
section. All cloud runs used single-GPU instances; no multi-node training.

## PerformanceRNN (LSTM) baseline

- **GPU:** local workstation, AMD GPU via ROCm.
- **Software:** PyTorch built against ROCm, run inside a ROCm container.
- **Precision:** fp32.
- **Notes:** matched-budget run mirroring the submitted v1 baseline
  (see the v1 report for the full baseline configuration).

## Aria fine-tunes (full-quality and real-time)

- **GPU:** 1 × NVIDIA B200 (≈180 GB VRAM, Blackwell).
- **Host:** server-class x86-64 CPU, ≈2 TB system RAM.
- **OS / stack:** Ubuntu 24.04, PyTorch 2.8 / CUDA 12.8.
- **Precision:** bf16 (`accelerate launch --mixed_precision bf16`).
- **Backbone:** Aria `medium` (d=1536, 16 layers, 24 heads, seq_len 8192);
  the real-time variant uses the embedding-projection (`medium-emb`) build so
  the fine-tuned weights drop into the real-time MLX deployment.
- **Wall-clock:** ≈3.5 h per variant for the full A→B→C→D fine-tune pipeline
  plus the Stage-C sampling sweep.
