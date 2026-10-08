# amd-ubuntu — Megatron-LM GPT-15.6B

Generated 2026-10-08 15:03 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 for RVS (ROCm Validation Suite) and rccl-tests, containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **GPT-15.6B on 8 GPUs (Megatron-LM v26.1 image, BF16, MBS 4, GBS 32): 578.2 TF/s/GPU on node6100 and 572.5 on node6101**, within 1% of each other.
- This is ≈27% below the 790.4 TF/s/GPU recorded earlier on Dell Cloud MI355X with the same model settings. Known differences: fused RoPE is turned off here (`--no-rope-fusion`) because it crashes on gfx950 in this image, and the host stack differs. The gap is worth investigating (e.g. a newer image with working fused RoPE) before using this number as a reference.

## Results

### megatron-ref GPT-15.6B (Megatron-LM v26.1), TF/s/GPU

| N | node6100 | node6101 | 6100/6101 |
|---|---:|---:|---:|
| 8 | 578.2 | 572.5 | 1.01x |

