# ROCm 10.1 vs ROCm 7.14 on MI355X

ROCm 10.1.0 (released 2026-10-05) against the current ROCm 7.14 results, on the same node (node6100, 8× MI355X; node6101 joins for the 2-node tests). Same scripts and settings; only the ROCm user-space stack differs. Higher is better unless the table says otherwise. Change = (10.1 − 7.14) / 7.14; within ±3% counts as about the same.

Plan: [`../plan-new-version.txt`](../plan-new-version.txt). Scripts: [`../rocm10.1/`](../rocm10.1/).

## Summary

- **hipBLASLt MX GEMM:** median +0.4% over 7 points; 1 faster, 0 slower, 6 about the same. Largest gain: MXBF8 (E5M2) (+3.6%).
- **RVS GEMM:** median 0.0% over 9 points; 1 faster, 0 slower, 8 about the same. Largest gain: fp4 8 GPUs (+11.1%).
- **rocHPL:** median 0.0% over 9 points; 0 faster, 0 slower, 9 about the same.
- **RCCL bandwidth, 1 node:** median 0.0% over 10 points; 0 faster, 0 slower, 10 about the same.
- **RCCL bandwidth, 2 nodes:** median -0.6% over 6 points; 0 faster, 0 slower, 6 about the same.
- **RCCL latency, 1 node:** median -1.1% over 21 points; 7 faster, 5 slower, 9 about the same. Largest gain: alltoall 1K (+36.8%). Largest loss: broadcast 1K (-34.5%).
- **Why 1-node latency changed:** real, not noise. RCCL 10.1 runs all_reduce, all_gather through a new DDA-IPC path; `RCCL_DDA_ENABLE=0` does not restore 7.14 (it is slower than both versions at 12 of 21 points), so DDA's share cannot be measured with that switch. sendrecv 1K, broadcast 1K: same with DDA on or off, a separate change.
- **RCCL latency, 2 nodes:** median 0.0% over 21 points; 6 faster, 3 slower, 12 about the same. Largest gain: all_reduce 1M (+13.7%). Largest loss: all_reduce 8 (-41.6%).
- **Why 2-node all_reduce differs:** So the 2-node default problem on 10.1 is RCCL 2.30.7's Tree/LL path across nodes: 7.14 runs the same algorithm and protocol correctly and fast. This is a regression in RCCL, not a setting in this benchmark. Same setting on both (`NCCL_PROTO=LL128`), 10.1 vs 7.14: 8 -2.6%; 1K -2.1%; 8K -3.6%; 64K -3.2%; 256K -5.1%; 1M -17.4%. 1 sizes differ by more than 10% even on the same setting.
- **RCCL latency, 2 nodes, 1 GPU/node:** median -2.8% over 8 points; 0 faster, 3 slower, 5 about the same. Largest loss: all_reduce 8 (-6.4%).
- **MX correctness:** every MX case passed on both versions.
- **Not affected, not rerun:** network-only tests, Primus training and Kimi-K3 inference.

## Versions

| | Current | New |
|---|---|---|
| ROCm | 7.14.0 | 10.1.0 |
| HIP | 7.14.60850-0000000 | 7.16.26385-0000000 |
| hipBLASLt | 1.4.1 | 1.4.1 (same version) |
| RCCL | 2.30.4 | 2.30.7 |
| rocBLAS | 5.5.0 | 5.7.0 |
| AMD clang | 23.0.0git | 24.0.0git |
| amdgpu driver (host, not changed) | 6.19.14.31400100 | 6.19.14.31400100 |
| RVS | 1.7.8, commit c82e4aeb | same commit, built against 10.1 |
| rocHPL / Open MPI / UCX | rocHPL d76d0c3, Open MPI 5.0.8, UCX 1.18.0 | same versions, rebuilt against 10.1 |

hipBLASLt reports the same version (1.4.1) in both stacks: ROCm 7.14 already shipped hipBLASLt 1.4.1. Its kernels and the rest of the stack (HIP, compiler, rocBLAS) still changed, so the GEMM results can still differ.

## What the update affects

| Result | Uses | Affected? | Rerun here |
|---|---|---|---|
| RVS GEMM, all precisions (§1, §3 of `mi355x-vs-b200.md`) | RVS `gst` → hipBLASLt | Yes | Yes |
| hipBLASLt MX GEMM correctness (MXFP8, MXFP4) | hipBLASLt | Yes | Yes (both stacks) |
| rocHPL FP64 (§2, `rochpl-vs-rvs.md`) | rocBLAS + HIP + Open MPI/UCX | Yes | Yes |
| RCCL bandwidth and latency, 1 and 2 nodes (§4, interconnect and latency files) | RCCL | Yes | Yes |
| Network alone (`ib_write_bw`, `ib_lat`) | perftest, host memory, no ROCm | No | No |
| Training, Primus / Megatron-LM (§5) | container with its own ROCm | No | No |
| Inference, Kimi-K3 (§6, Kimi files) | containers with their own ROCm | No | No |

## 1. hipBLASLt MX GEMM correctness (MXFP8, MXFP6, MXFP4)

`hipblaslt-bench -v` (result checked against a CPU reference), AMD's own bench from each release's gfx950 test tarball, 1 GPU. Scale mode 3 = MX block scaling (one E8M0 scale per 32 elements). Check at 2048×2048×2048 (PASS = error within the bench's own tolerance); speed at 8192×8192×8192.

The MX fixes in the 10.1 release notes are in Composable Kernel: most name gfx1250, and one fixes MX GEMMs where B is packed more densely than A (such as FP8 × FP4). The FP8 × FP4 and FP4 × FP8 rows test that case on gfx950.

| Case | A × B | 7.14 check (error ≤ limit) | 7.14 TFLOPS | 10.1 check | 10.1 TFLOPS | Change |
|---|---|---|---:|---|---:|---:|
| MXFP8 (E4M3) | f8_r × f8_r | PASS (0.0036 ≤ 0.1) | 2,040.0 | PASS (0.0036 ≤ 0.1) | 2,083.6 | +2.1% |
| MXBF8 (E5M2) | bf8_r × bf8_r | PASS (0.0037 ≤ 0.01) | 2,039.8 | PASS (0.0037 ≤ 0.01) | 2,113.7 | +3.6% |
| MXFP6 (E2M3) | f6_r × f6_r | PASS (0.0035 ≤ 0.01) | 1,250.8 | PASS (0.0035 ≤ 0.01) | 1,250.9 | 0.0% |
| MXBF6 (E3M2) | bf6_r × bf6_r | PASS (0.0037 ≤ 0.01) | 1,250.6 | PASS (0.0037 ≤ 0.01) | 1,255.2 | +0.4% |
| MXFP4 (E2M1) | f4_r × f4_r | PASS (0.004 ≤ 0.01) | 3,364.9 | PASS (0.004 ≤ 0.01) | 3,376.5 | +0.3% |
| FP8 x FP4 | f8_r × f4_r | PASS (0.0038 ≤ 0.01) | 2,241.2 | PASS (0.0038 ≤ 0.01) | 2,209.6 | -1.4% |
| FP4 x FP8 | f4_r × f8_r | PASS (0.0038 ≤ 0.01) | 2,285.0 | PASS (0.0038 ≤ 0.01) | 2,298.9 | +0.6% |

## 2. RVS GEMM (hipBLASLt), TFLOPS

Same RVS `gst` settings as §1/§3 of `mi355x-vs-b200.md` (level-5 matrix sizes, 30 s per test, peak of the 3 s samples). FP4, FP6 and FP8 are the MX formats.

| Precision | 8 GPUs, 7.14 | 8 GPUs, 10.1 | Change | 1 GPU, 7.14 | 1 GPU, 10.1 | Change |
|---|---:|---:|---:|---:|---:|---:|
| FP4 | 17,027.5 | 18,913.5 | +11.1% | 4,079.3 | 4,079.9 | 0.0% |
| FP6 | 9,925.1 | 9,930.5 | +0.1% | 1,240.5 | 1,238.0 | -0.2% |
| BF6 | 9,914.4 | 9,925.1 | +0.1% | 1,240.9 | 1,240.4 | 0.0% |
| FP8 | 30,278.5 | 30,170.3 | -0.4% | 3,752.6 | 3,628.8 | -3.3% |
| BF8 | 27,282.0 | 27,065.9 | -0.8% | 3,377.9 | 3,388.3 | +0.3% |
| FP16 | 12,764.0 | 12,716.6 | -0.4% | 1,570.5 | 1,582.0 | +0.7% |
| BF16 | 13,558.4 | 13,549.2 | -0.1% | 1,670.7 | 1,668.6 | -0.1% |
| FP32 | 1,230.9 | 1,230.7 | 0.0% | 152.6 | 152.5 | -0.1% |
| FP64 | 616.1 | 616.2 | 0.0% | 76.5 | 76.5 | 0.0% |

All 72 points (1–8 GPUs × precisions): median change +0.0%, range -7.0% to +23.9%.

## 3. rocHPL FP64 Linpack, TFLOPS

Same rocHPL commit, N, NB (512) and process grids as the ROCm 7.14 run (`rochpl_20261020_204342`, reported in `rochpl-vs-rvs.md`). Each run must pass HPL's residual check.

| GPUs | P×Q | N (7.14 / 10.1) | 7.14 | 10.1 | Change | Passed (7.14 / 10.1) |
|---:|---|---|---:|---:|---:|---|
| 1 | 1×1 | 163840 / 163840 | 60.1 | 60.1 | 0.0% | yes / yes |
| 2 | 1×2 | 231424 / 231424 | 124.1 | 124.4 | +0.2% | yes / yes |
| 3 | 1×3 | 284160 / 284160 | 188.6 | 188.6 | 0.0% | yes / yes |
| 4 | 2×2 | 327680 / 327680 | 251.2 | 251.7 | +0.2% | yes / yes |
| 5 | 1×5 | 366080 / 366080 | 318.6 | 318.6 | 0.0% | yes / yes |
| 6 | 2×3 | 402432 / 402432 | 383.9 | 384.2 | +0.1% | yes / yes |
| 7 | 1×7 | 433664 / 433664 | 448.1 | 448.3 | 0.0% | yes / yes |
| 8 | 2×4 | 462848 / 462848 | 515.4 | 515.6 | 0.0% | yes / yes |
| 16 | 4×4 | 655360 / 655360 | 976.7 | 975.4 | -0.1% | yes / yes |

## 4. RCCL bandwidth (bus bandwidth, GB/s)

### 4.1 One node, 8 GPUs

rccl-tests, default RCCL settings, peak bus bandwidth over message sizes up to 8 GB.

| Collective | 7.14 | 10.1 | Change |
|---|---:|---:|---:|
| all_gather | 388.6 | 389.5 | +0.2% |
| all_reduce | 396.1 | 396.1 | 0.0% |
| alltoall | 347.6 | 347.2 | -0.1% |
| alltoallv | 218.1 | 217.8 | -0.2% |
| broadcast | 389.1 | 388.3 | -0.2% |
| gather | 448.6 | 448.2 | -0.1% |
| reduce | 349.6 | 351.9 | +0.6% |
| reduce_scatter | 409.4 | 406.8 | -0.6% |
| scatter | 432.8 | 436.9 | +0.9% |
| sendrecv | 60.7 | 60.8 | +0.1% |

### 4.2 Two nodes, 16 GPUs

rccl-tests over the 8 RoCE rails (RCCL NET/IB), 8 GPUs per node, peak bus bandwidth and bus bandwidth at 1 GB.

| Collective | Peak, 7.14 | Peak, 10.1 | Change | 1 GB, 7.14 | 1 GB, 10.1 | Change |
|---|---:|---:|---:|---:|---:|---:|
| all_gather | 380.0 | 376.8 | -0.8% | 380.0 | 375.5 | -1.2% |
| all_reduce | 384.6 | 382.6 | -0.5% | 384.6 | 381.0 | -0.9% |
| alltoall | 77.8 | 77.2 | -0.7% | 77.4 | 76.7 | -0.9% |
| broadcast | 364.6 | 360.0 | -1.3% | 331.6 | 326.1 | -1.6% |
| reduce_scatter | 372.2 | 372.4 | 0.0% | 372.2 | 371.5 | -0.2% |
| sendrecv | 32.1 | 32.4 | +1.0% | 32.0 | 32.2 | +0.6% |

## 5. RCCL latency (time per call, µs, lower is better)

rccl-tests, 1000 iterations + 100 warm-up, no data check. In the Change column a positive number means 10.1 is faster.

### 5.1 One node, 8 GPUs (median of the three default runs)

| Collective | Size | 7.14 (µs) | 10.1 (µs) | Change (lower time = positive) |
|---|---:|---:|---:|---:|
| sendrecv | 8 | 6.1 | 7.0 | -14.8% |
| sendrecv | 1K | 6.5 | 8.4 | -29.2% |
| sendrecv | 64K | 13.4 | 13.9 | -3.7% |
| sendrecv | 1M | 29.6 | 30.0 | -1.4% |
| all_reduce | 8 | 13.9 | 15.9 | -14.4% |
| all_reduce | 1K | 5.0 | 5.1 | -2.0% |
| all_reduce | 64K | 6.5 | 6.4 | +1.5% |
| all_reduce | 1M | 17.4 | 13.8 | +20.7% |
| all_gather | 1K | 7.8 | 5.1 | +34.6% |
| all_gather | 64K | 8.8 | 6.4 | +27.3% |
| all_gather | 1M | 11.7 | 7.8 | +33.3% |
| reduce_scatter | 1K | 6.1 | 6.2 | -1.6% |
| reduce_scatter | 64K | 7.1 | 7.0 | +1.4% |
| reduce_scatter | 1M | 8.7 | 8.8 | -1.1% |
| broadcast | 8 | 6.1 | 6.1 | 0.0% |
| broadcast | 1K | 5.8 | 7.8 | -34.5% |
| broadcast | 64K | 15.2 | 15.6 | -2.6% |
| broadcast | 1M | 41.7 | 42.6 | -2.2% |
| alltoall | 1K | 7.6 | 4.8 | +36.8% |
| alltoall | 64K | 10.4 | 7.6 | +26.9% |
| alltoall | 1M | 14.0 | 9.2 | +34.3% |

#### 5.1.1 Why the 1-node numbers changed

Turning DDA off does not bring back 7.14's behaviour. With `RCCL_DDA_ENABLE=0`, 12 of 21 points are more than 10% slower than both 7.14 and 10.1's default, in all_reduce, all_gather, reduce_scatter, broadcast, alltoall. So the switch removes DDA and leaves a slower fallback path, not 7.14's path, and this test cannot measure how much of the change DDA explains. 10.1 runs all_reduce, all_gather through the new DDA-IPC implementation (debug log, §2); the DDA switch also changes reduce_scatter, broadcast, alltoall, which log no DDA line. The faster points come with this rework of RCCL's single-node path in 2.30.7 (all_reduce 1M, all_gather 1K, all_gather 64K, all_gather 1M, alltoall 1K, alltoall 64K, alltoall 1M). sendrecv 1K, broadcast 1K change by the same amount with DDA on or off, so that change is separate from DDA (another change in RCCL 2.30.7 / HIP 7.16). Details, same-day timing and RCCL's choices: [`rccl-lat-1node-rocm10.1-why.md`](rccl-lat-1node-rocm10.1-why.md).

| Collective | Size | 7.14 (µs) | 10.1 (µs) | 10.1, DDA off (µs) |
|---|---:|---:|---:|---:|
| sendrecv | 1K | 6.5 | 8.4 | 8.5 |
| all_reduce | 1M | 17.4 | 13.7 | 35.1 |
| all_gather | 1K | 7.8 | 5.0 | 13.4 |
| all_gather | 64K | 9.0 | 6.3 | 14.7 |
| all_gather | 1M | 11.7 | 7.9 | 30.5 |
| broadcast | 1K | 5.8 | 7.7 | 8.0 |
| alltoall | 1K | 7.5 | 4.9 | 9.2 |
| alltoall | 64K | 10.4 | 7.6 | 11.2 |
| alltoall | 1M | 14.0 | 9.2 | 15.7 |

On ROCm 10.1, RCCL 2.30.7's default all_reduce across 2 nodes x 8 GPUs gives wrong results (rccl-tests data check) and takes 1.4-2.2 ms per call from 8 B to 256 KB. `NCCL_PROTO=LL128` gives correct results at every size, so the 10.1 all_reduce rows in 5.2 use it; the other 10.1 rows and all 7.14 rows use RCCL's default (all settings: `node6100/rocm10.1/fix_check.md`, `node6100/rocm10.1/rccl_lat.md`).

### 5.2 Two nodes, 8 GPUs per node (default setting; 10.1 all_reduce with `NCCL_PROTO=LL128`)

| Collective | Size | 7.14 (µs) | 10.1 (µs) | Change (lower time = positive) |
|---|---:|---:|---:|---:|
| sendrecv | 8 | 20.7 | 18.7 | +9.7% |
| sendrecv | 1K | 20.4 | 18.9 | +7.4% |
| sendrecv | 64K | 23.2 | 21.9 | +5.6% |
| sendrecv | 1M | 49.1 | 49.5 | -0.8% |
| all_reduce | 8 | 27.9 | 39.5 | -41.6% |
| all_reduce | 1K | 30.6 | 39.6 | -29.4% |
| all_reduce | 64K | 41.9 | 45.4 | -8.4% |
| all_reduce | 1M | 66.5 | 57.4 | +13.7% |
| all_gather | 1K | 36.1 | 36.7 | -1.7% |
| all_gather | 64K | 38.2 | 38.7 | -1.3% |
| all_gather | 1M | 54.5 | 54.6 | -0.2% |
| reduce_scatter | 1K | 35.6 | 35.6 | 0.0% |
| reduce_scatter | 64K | 43.3 | 43.3 | 0.0% |
| reduce_scatter | 1M | 55.0 | 54.7 | +0.5% |
| broadcast | 8 | 15.9 | 15.2 | +4.4% |
| broadcast | 1K | 10.1 | 10.1 | 0.0% |
| broadcast | 64K | 45.5 | 45.2 | +0.7% |
| broadcast | 1M | 60.1 | 61.6 | -2.5% |
| alltoall | 1K | 36.9 | 35.2 | +4.6% |
| alltoall | 64K | 36.6 | 36.4 | +0.5% |
| alltoall | 1M | 46.6 | 45.3 | +2.8% |

#### 5.2.1 Why 2-node all_reduce differs

- Both versions pick the same algorithm and protocol by default (8: TREE/LL; 1K: TREE/LL; 8K: TREE/LL; 64K: TREE/LL; 256K: TREE/LL; 1M: TREE/LL128).
- Default setting, data check: wrong results on 7.14 at no size, on 10.1 at 16K, 32K, 128K, 256K.
- Tree/LL forced on both: wrong results on 7.14 at no size, on 10.1 at 256K, 512K; 10.1 is more than 3× slower at 8, 1K, 8K, 64K, 256K (e.g. 8: 28.7 vs 1,426 µs).
- So the 2-node default problem on 10.1 is RCCL 2.30.7's Tree/LL path across nodes: 7.14 runs the same algorithm and protocol correctly and fast. This is a regression in RCCL, not a setting in this benchmark.
- Same setting on both (`NCCL_PROTO=LL128`), 10.1 vs 7.14: 8 -2.6%; 1K -2.1%; 8K -3.6%; 64K -3.2%; 256K -5.1%; 1M -17.4%. 1 sizes differ by more than 10% even on the same setting.
- On 7.14 itself, LL128 vs its default: 8 +43.7%; 1K +31.2%; 8K +37.3%; 64K +11.9%; 256K -0.6%; 1M +5.3% (positive = LL128 slower).
- 10.1 with `RCCL_DDA_ENABLE=0`: wrong results at 64K; 8 B 1,416 µs vs 1,416 µs default, so DDA is not behind the 2-node problem.

Same-day runs with the same settings on both versions, correctness checks and RCCL's choices: [`rccl-lat-2node-rocm10.1-why.md`](rccl-lat-2node-rocm10.1-why.md).

### 5.3 Two nodes, 1 GPU per node (default setting)

| Collective | Size | 7.14 (µs) | 10.1 (µs) | Change (lower time = positive) |
|---|---:|---:|---:|---:|
| sendrecv | 8 | 27.4 | 27.1 | +1.1% |
| sendrecv | 1K | 27.8 | 28.6 | -2.9% |
| sendrecv | 64K | 37.0 | 38.0 | -2.7% |
| sendrecv | 1M | 63.1 | 62.5 | +1.0% |
| all_reduce | 8 | 20.4 | 21.7 | -6.4% |
| all_reduce | 1K | 21.0 | 21.7 | -3.3% |
| all_reduce | 64K | 36.9 | 38.7 | -4.9% |
| all_reduce | 1M | 67.7 | 67.3 | +0.6% |

