# amd-ubuntu — RVS (ROCm Validation Suite) gst TFLOPS

Generated 2026-10-08 15:30 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 for RVS and rccl-tests, containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **Both nodes healthy and matched.** Every precision is within 3% between node6100 and node6101; fp32/fp64 are identical to 1%.
- **Scaling is linear on host ROCm 7.2.4**: N=8 gives 99–104% of 8 × N=1 for all nine precisions, so there is no node-level power or thermal limit when all 8 GPUs run GEMMs at once.
- Headline N=8 aggregate (node6100): fp8 30.3 PF, bf8 26.9 PF, fp4 25.6 PF, bf16 13.6 PF, fp64 617 TF.
- fp4 differs between ROCm 7.2.4 and 7.14; explained in `../vs-amd-cloud/rvs.md`.

## Results

Aggregate `gst` TFLOPS: hipBLASLt GEMM on every selected GPU at once, peak per-GPU sample summed over GPUs. Each GPU runs its own GEMM, so scaling should be linear; losses are power or thermals.

### N = 8

| precision | node6100 | node6101 | 6100/6101 |
|---|---:|---:|---:|
| bf16 | 13,556.5 | 13,191.2 | 1.03x |
| bf6 | 10,264.9 | 10,268.3 | 1.00x |
| bf8 | 26,948.0 | 26,861.6 | 1.00x |
| fp16 | 12,703.1 | 12,442.3 | 1.02x |
| fp32 | 1,231.6 | 1,229.5 | 1.00x |
| fp4 | 25,596.3 | 25,530.0 | 1.00x |
| fp6 | 10,266.3 | 10,267.8 | 1.00x |
| fp64 | 616.6 | 615.6 | 1.00x |
| fp8 | 30,275.3 | 29,690.4 | 1.02x |

### N = 1

| precision | node6100 | node6101 | 6100/6101 |
|---|---:|---:|---:|
| bf16 | 1,672.2 | 1,617.2 | 1.03x |
| bf6 | 1,282.2 | 1,285.5 | 1.00x |
| bf8 | 3,393.6 | 3,321.8 | 1.02x |
| fp16 | 1,586.6 | 1,558.5 | 1.02x |
| fp32 | 152.4 | 154.3 | 0.99x |
| fp4 | 3,187.3 | 3,175.0 | 1.00x |
| fp6 | 1,282.0 | 1,286.5 | 1.00x |
| fp64 | 76.7 | 77.3 | 0.99x |
| fp8 | 3,683.5 | 3,558.3 | 1.04x |

### Scaling efficiency, N = 8 vs 8 × N = 1 (%)

| precision | node6100 | node6101 |
|---|---:|---:|
| bf16 | 101.3 | 102.0 |
| bf6 | 100.1 | 99.8 |
| bf8 | 99.3 | 101.1 |
| fp16 | 100.1 | 99.8 |
| fp32 | 101.0 | 99.6 |
| fp4 | 100.4 | 100.5 |
| fp6 | 100.1 | 99.8 |
| fp64 | 100.6 | 99.5 |
| fp8 | 102.7 | 104.3 |
