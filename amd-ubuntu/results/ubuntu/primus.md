# amd-ubuntu — Primus: Megatron-LM llama2-7B and GEMM microbench

Generated 2026-10-05 18:26 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 (RVS, rccl-tests), containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **GEMM microbench**: 1,400–1,510 TF/s per GPU at every N, flat from 1 to 8 GPUs (independent GEMMs), with node6100 ≈3–5% above node6101, consistent with RVS.
- **Megatron llama2-7B**: the first run (2026-10-02) failed at every N on both nodes because a compiled extension build ran out of space on the container's 64 MB scratch area. It is being rerun with a persistent per-node overlay (`primus/rerun_megatron_llama.sh`); the table below fills in when it finishes.

## Results

### Primus GEMM microbench, mean TF/s/GPU

| N | node6100 | node6101 | 6100/6101 |
|---|---:|---:|---:|
| 1 | 1,480.5 | 1,425.4 | 1.04x |
| 2 | 1,479.9 | 1,401.3 | **1.06x** |
| 3 | 1,499.1 | 1,424.0 | **1.05x** |
| 4 | 1,499.8 | 1,430.4 | 1.05x |
| 5 | 1,505.9 | 1,444.6 | 1.04x |
| 6 | 1,511.0 | 1,461.8 | 1.03x |
| 7 | 1,509.8 | 1,463.3 | 1.03x |
| 8 | 1,504.2 | 1,468.0 | 1.02x |

