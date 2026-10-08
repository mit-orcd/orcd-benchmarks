# amd-ubuntu — Primus: Megatron-LM llama2-7B and GEMM microbench

Generated 2026-10-08 15:03 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 for RVS (ROCm Validation Suite) and rccl-tests, containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **Megatron llama2-7B BF16 (Primus v26.5): 1,066–1,178 compute TF/s/GPU from 1 to 8 GPUs**, about 1,135–1,170 at 8 GPUs. Iteration time stays ≈4.8–5.3 s at every N (weak scaling, constant work per GPU), so single-node data-parallel scaling is near-perfect. A small dip at N=5–7 (≈1,070–1,110) matches the RCCL partial-mesh dip in `rccl.md`; use 1, 2, 4 or 8 GPUs.
- node6100 is ≈2–4% faster than node6101 at every N, the same offset seen in RVS and the GEMM microbench.
- Wall-clock TF/s (≈170–270) is far below compute TF/s because it includes container start, pip installs and, on the first run, building the apex extension (N=1 took 32 min of which most was the build). Use compute TF/s for comparisons.
- **GEMM microbench**: 1,400–1,510 TF/s per GPU at every N, flat from 1 to 8 GPUs (independent GEMMs).
- History: the first run (2026-10-02) failed at every N because the extension build ran out of space in the 64 MB container scratch area; rerun 2026-10-05 with a persistent per-node overlay (`primus/rerun_megatron_llama.sh`), all 16 runs OK.

## Results

### Megatron-LM llama2-7B BF16 via Primus, compute TF/s/GPU

| N | node6100 | node6101 | 6100/6101 |
|---|---:|---:|---:|
| 1 | 1,177.9 | 1,154.6 | 1.02x |
| 2 | 1,152.7 | 1,110.0 | 1.04x |
| 3 | 1,156.0 | 1,114.0 | 1.04x |
| 4 | 1,170.5 | 1,125.8 | 1.04x |
| 5 | 1,113.7 | 1,075.7 | 1.04x |
| 6 | 1,110.3 | 1,071.8 | 1.04x |
| 7 | 1,107.4 | 1,065.9 | 1.04x |
| 8 | 1,170.5 | 1,134.8 | 1.03x |

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

