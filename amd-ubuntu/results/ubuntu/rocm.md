# amd-ubuntu — Host ROCm 7.2.4 vs ROCm 7.14 (RVS, RCCL)

Generated 2026-10-05 18:26 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 (RVS, rccl-tests), containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **Same driver, two user spaces: ROCm 7.14 matches 7.2.4 within ±3% for nearly everything** (RVS bf16/fp16/fp8/fp32/fp64, single-node RCCL at N=8).
- **fp4 is the exception.** On 7.14 a single GPU is 1.26–1.28× faster (≈4,080 vs 3,190 TFLOPS), but with all 8 GPUs at once the per-GPU rate falls to ≈2,130, so the 8-GPU aggregate is 0.67× of 7.2.4. On 7.2.4 fp4 scales linearly. The same 8-GPU fall-off appears in the amd-cloud run (also 7.14), so it comes with the 7.14 fp4 path, not with these nodes.
- bf6/fp6 are ≈3% lower on 7.14.
- **N=5 RCCL**: 7.14 fixes alltoallv (≈45 vs ≈22 GB/s on 7.2.4).
- **2-node RCCL**: all_reduce at PPN=8 is ≈381–385 GB/s on both stacks; alltoall is ≈14% lower on 7.14 (77 vs 90 GB/s); sendrecv slightly higher on 7.14 (32 vs 29).
- Recommendation: either stack is fine for training; prefer 7.2.4 for 8-GPU fp4 GEMM work until the 7.14 fp4 scaling is understood.

## Results

Same nodes, same amdgpu driver (6.19.14), same benchmark scripts; only the user-space ROCm differs: host ROCm 7.2.4 (`/opt/rocm`, packaged RVS) vs ROCm 7.14.0 (TheRock tarball in `amd-software/rocm-7.14.0`, RVS built from source). Ratio = 7.14 / 7.2.4; bold = more than 5%.

### RVS gst aggregate TFLOPS

#### N = 8

| precision | 6100 7.2.4 | 6100 7.14 | 6101 7.2.4 | 6101 7.14 | 6100 7.14/7.2.4 | 6101 7.14/7.2.4 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 13,556.5 | 13,558.4 | 13,191.2 | 13,200.5 | 1.00x | 1.00x |
| bf6 | 10,264.9 | 9,914.4 | 10,268.3 | 9,915.2 | 0.97x | 0.97x |
| bf8 | 26,948.0 | 27,282.0 | 26,861.6 | 26,600.0 | 1.01x | 0.99x |
| fp16 | 12,703.1 | 12,764.0 | 12,442.3 | 12,484.6 | 1.00x | 1.00x |
| fp32 | 1,231.6 | 1,230.9 | 1,229.5 | 1,228.8 | 1.00x | 1.00x |
| fp4 | 25,596.3 | 17,027.5 | 25,530.0 | 17,099.5 | **0.67x** | **0.67x** |
| fp6 | 10,266.3 | 9,925.1 | 10,267.8 | 9,913.7 | 0.97x | 0.97x |
| fp64 | 616.6 | 616.1 | 615.6 | 615.0 | 1.00x | 1.00x |
| fp8 | 30,275.3 | 30,278.5 | 29,690.4 | 29,312.9 | 1.00x | 0.99x |

#### N = 1

| precision | 6100 7.2.4 | 6100 7.14 | 6101 7.2.4 | 6101 7.14 | 6100 7.14/7.2.4 | 6101 7.14/7.2.4 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 1,672.2 | 1,670.7 | 1,617.2 | 1,631.1 | 1.00x | 1.01x |
| bf6 | 1,282.2 | 1,240.9 | 1,285.5 | 1,240.9 | 0.97x | 0.97x |
| bf8 | 3,393.6 | 3,377.9 | 3,321.8 | 3,342.0 | 1.00x | 1.01x |
| fp16 | 1,586.6 | 1,570.5 | 1,558.5 | 1,519.6 | 0.99x | 0.98x |
| fp32 | 152.4 | 152.6 | 154.3 | 154.1 | 1.00x | 1.00x |
| fp4 | 3,187.3 | 4,079.3 | 3,175.0 | 3,989.8 | **1.28x** | **1.26x** |
| fp6 | 1,282.0 | 1,240.5 | 1,286.5 | 1,240.9 | 0.97x | 0.96x |
| fp64 | 76.7 | 76.5 | 77.3 | 77.2 | 1.00x | 1.00x |
| fp8 | 3,683.5 | 3,752.6 | 3,558.3 | 3,628.8 | 1.02x | 1.02x |

### RCCL busbw at the largest size (GB/s), single node

#### N = 8

| collective | 6100 7.2.4 | 6100 7.14 | 6101 7.2.4 | 6101 7.14 | 6100 7.14/7.2.4 | 6101 7.14/7.2.4 |
|---|---:|---:|---:|---:|---:|---:|
| all_gather | 388.4 | 389.0 | 387.7 | 389.7 | 1.00x | 1.01x |
| all_reduce | 394.4 | 395.8 | 393.1 | 398.0 | 1.00x | 1.01x |
| alltoall | 343.1 | 347.6 | 342.6 | 347.5 | 1.01x | 1.01x |
| alltoallv | 216.8 | 218.1 | 217.0 | 220.4 | 1.01x | 1.02x |
| broadcast | 386.7 | 389.1 | 387.1 | 389.7 | 1.01x | 1.01x |
| gather | 425.4 | 424.8 | 424.7 | 422.8 | 1.00x | 1.00x |
| reduce | 327.4 | 322.0 | 329.2 | 328.6 | 0.98x | 1.00x |
| reduce_scatter | 386.3 | 388.4 | 386.6 | 388.9 | 1.01x | 1.01x |
| scatter | 397.1 | 395.6 | 397.1 | 396.1 | 1.00x | 1.00x |
| sendrecv | 60.7 | 60.3 | 60.2 | 60.5 | 0.99x | 1.01x |

#### N = 5

| collective | 6100 7.2.4 | 6100 7.14 | 6101 7.2.4 | 6101 7.14 | 6100 7.14/7.2.4 | 6101 7.14/7.2.4 |
|---|---:|---:|---:|---:|---:|---:|
| all_gather | 46.0 | 45.9 | 45.9 | 46.0 | 1.00x | 1.00x |
| all_reduce | 47.1 | 48.1 | 47.2 | 48.2 | 1.02x | 1.02x |
| alltoall | 61.0 | 61.5 | 60.8 | 61.2 | 1.01x | 1.01x |
| alltoallv | 22.0 | 45.4 | 23.0 | 45.2 | **2.07x** | **1.96x** |
| broadcast | 41.9 | 39.9 | 41.7 | 40.0 | 0.95x | 0.96x |
| gather | 70.4 | 73.2 | 70.2 | 70.7 | 1.04x | 1.01x |
| reduce | 48.6 | 46.4 | 49.5 | 45.8 | 0.95x | **0.93x** |
| reduce_scatter | 47.6 | 46.4 | 47.9 | 46.7 | 0.97x | 0.97x |
| scatter | 71.1 | 76.3 | 69.8 | 75.7 | **1.07x** | **1.08x** |
| sendrecv | 61.0 | 60.5 | 60.5 | 60.1 | 0.99x | 0.99x |

### RCCL busbw at the largest size (GB/s), 2 nodes

#### all PPN

| collective | ppn | 6100 7.2.4 | 6100 7.14 | 6100 7.14/7.2.4 |
|---|---|---:|---:|---:|
| all_gather | 1 | 48.3 | 40.4 | **0.84x** |
| all_gather | 2 | 48.0 | 48.9 | 1.02x |
| all_gather | 4 | 148.1 | 175.5 | **1.19x** |
| all_gather | 8 | 376.7 | 372.3 | 0.99x |
| all_reduce | 1 | 48.7 | 48.5 | 1.00x |
| all_reduce | 2 | 48.7 | 48.9 | 1.00x |
| all_reduce | 4 | 159.8 | 177.4 | **1.11x** |
| all_reduce | 8 | 380.9 | 372.6 | 0.98x |
| alltoall | 8 | 89.6 | 77.1 | **0.86x** |
| broadcast | 8 | 369.9 | 364.6 | 0.99x |
| reduce_scatter | 8 | 378.5 | 371.6 | 0.98x |
| sendrecv | 8 | 28.6 | 32.1 | **1.12x** |

