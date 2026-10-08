# RVS (ROCm Validation Suite) fp4 recheck: ROCm 7.14 vs 7.2.4, 1 vs 8 GPUs

Run: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/rvs/fp4_recheck_20261007_201109` (script `work-rocmval/fp4_recheck_and_profile.sh`). gst fp4, the sweep's own configs (8192×8192×16384, 30 s), peak per-GPU sample like the sweep.

**Verdict: CONFIRMED 7.14: n1=3984 n8=2284 (0.57x) m8=4029 (1.01x)**

## 1. Recheck: per-GPU fp4 TFLOPS (mean over repeats)

| arm | ROCm 7.14 per GPU | 7.14 vs its 1 GPU | 7.14 slowest–fastest GPU | ROCm 7.2.4 per GPU | 7.2.4 vs its 1 GPU | 7.2.4 slowest–fastest GPU | 7.14 / 7.2.4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 GPU | 3,984 | 1.00x | 3,977–3,988 | 3,185 | 1.00x | 3,184–3,187 | 1.25x |
| 2 GPUs, one process | 3,941 | 0.99x | 3,888–3,990 | 3,159 | 0.99x | 3,133–3,184 | 1.25x |
| 4 GPUs, one process | 3,295 | 0.83x | 2,218–4,068 | 3,176 | 1.00x | 3,132–3,198 | 1.04x |
| 8 GPUs, one process (standard) | 2,284 | 0.57x | 1,374–3,463 | 3,192 | 1.00x | 3,132–3,227 | 0.72x |
| 8 GPUs, 8 processes | 4,029 | 1.01x | 3,897–4,086 | 3,190 | 1.00x | 3,122–3,228 | 1.26x |

3 repeats per cell. Per-run numbers: `stage1.txt`.

### Clocks, power and host CPU during the runs (first repeat)

| arm | stack | mean sclk MHz | mean power W/GPU | rvs CPU % (all threads) | busiest rvs thread % |
|---|---|---:|---:|---:|---:|
| 1 GPU | ROCm 7.14 | 1,629 | 264 | — | — |
| 1 GPU | ROCm 7.2.4 | 2,102 | 322 | — | — |
| 2 GPUs, one process | ROCm 7.14 | 1,963 | 417 | — | — |
| 2 GPUs, one process | ROCm 7.2.4 | 2,045 | 389 | — | — |
| 4 GPUs, one process | ROCm 7.14 | 1,999 | 476 | — | — |
| 4 GPUs, one process | ROCm 7.2.4 | 2,014 | 489 | — | — |
| 8 GPUs, one process (standard) | ROCm 7.14 | 1,987 | 492 | — | — |
| 8 GPUs, one process (standard) | ROCm 7.2.4 | 1,952 | 644 | — | — |
| 8 GPUs, 8 processes | ROCm 7.14 | 1,907 | 762 | — | — |
| 8 GPUs, 8 processes | ROCm 7.2.4 | 2,100 | 795 | — | — |

## 2. Profile: GEMM kernels per GPU (rocprofv3 kernel trace, 10 s runs)

Kernel TFLOPS = one GEMM's FLOPs / its median duration (what the GPU does while the kernel runs). Busy = kernel time / wall time after the first 20% of the run; gap = median idle time between kernels. Slower kernels point at the device (clocks, a different kernel); same kernels with low busy / large gaps point at the host side.

### 7.14, 1 GPU

| agent | main kernel | count | median ms | kernel TFLOPS | busy | median gap µs |
|---|---|---:|---:|---:|---:|---:|
| Agent 2 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 24494 | 0.553 | 3,978 | 91% | 0.0 |

### 7.14, 8 GPUs one process

| agent | main kernel | count | median ms | kernel TFLOPS | busy | median gap µs |
|---|---|---:|---:|---:|---:|---:|
| Agent 2 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 25132 | 0.550 | 3,996 | 61% | 421.0 |
| Agent 3 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 11575 | 0.563 | 3,903 | 32% | 1,548.8 |
| Agent 4 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 25208 | 0.543 | 4,050 | 60% | 432.8 |
| Agent 5 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 11486 | 0.553 | 3,977 | 30% | 1,719.3 |
| Agent 6 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 25424 | 0.531 | 4,141 | 63% | 332.8 |
| Agent 7 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 11532 | 0.562 | 3,911 | 31% | 1,635.3 |
| Agent 8 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 11707 | 0.564 | 3,897 | 31% | 1,523.4 |
| Agent 9 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 11402 | 0.545 | 4,032 | 30% | 1,835.5 |

### 7.14, 8 processes

| agent | main kernel | count | median ms | kernel TFLOPS | busy | median gap µs |
|---|---|---:|---:|---:|---:|---:|
| Agent 2 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 26003 | 0.552 | 3,986 | 95% | 0.0 |
| Agent 3 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 25831 | 0.563 | 3,905 | 95% | 0.0 |
| Agent 4 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 25996 | 0.542 | 4,056 | 94% | 0.0 |
| Agent 5 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 26024 | 0.542 | 4,056 | 94% | 0.0 |
| Agent 6 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 26234 | 0.535 | 4,107 | 94% | 0.0 |
| Agent 7 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 26217 | 0.546 | 4,026 | 94% | 0.0 |
| Agent 8 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 26200 | 0.538 | 4,091 | 94% | 0.0 |
| Agent 9 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WG` | 26154 | 0.543 | 4,046 | 94% | 0.0 |

### 7.2.4, 8 GPUs one process

| agent | main kernel | count | median ms | kernel TFLOPS | busy | median gap µs |
|---|---|---:|---:|---:|---:|---:|
| Agent 2 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_B_SB_B_WGT_256x256x128_UR_4_WGM_` | 24796 | 0.692 | 3,179 | 95% | 0.0 |
| Agent 3 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_B_SB_B_WGT_256x256x128_UR_4_WGM_` | 24969 | 0.697 | 3,155 | 96% | 0.0 |
| Agent 4 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_B_SB_B_WGT_256x256x128_UR_4_WGM_` | 25185 | 0.685 | 3,211 | 97% | 0.0 |
| Agent 5 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_B_SB_B_WGT_256x256x128_UR_4_WGM_` | 25076 | 0.690 | 3,186 | 97% | 0.0 |
| Agent 6 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_B_SB_B_WGT_256x256x128_UR_4_WGM_` | 25043 | 0.686 | 3,203 | 96% | 0.0 |
| Agent 7 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_B_SB_B_WGT_256x256x128_UR_4_WGM_` | 25138 | 0.684 | 3,214 | 96% | 0.0 |
| Agent 8 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_B_SB_B_WGT_256x256x128_UR_4_WGM_` | 24888 | 0.688 | 3,194 | 93% | 0.0 |
| Agent 9 | `RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_B_SB_B_WGT_256x256x128_UR_4_WGM_` | 25295 | 0.690 | 3,187 | 97% | 0.0 |

### Reading

- 7.14, 8 GPUs one process vs 1 GPU: kernel duration 0.96–1.02× the 1-GPU kernel; busy 30–63% (1 GPU: 91%); same GEMM kernel as on 1 GPU.
- The GPUs sit idle between kernels → host-side cause: the single rvs process does not launch work fast enough for all 8 GPUs (launch/synchronisation contention).

