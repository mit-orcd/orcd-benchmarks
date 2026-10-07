- **Same ROCm 7.14 on both sides (apple-to-apple): amd-ubuntu matches amd-cloud within 0–5%** for every precision at N=1 and N=8 (fp4 0.96–1.03×, all others 1.00–1.05×). The hardware performs the same.
- node6100 runs bf16/fp16/fp8 ≈2–3% faster than node6101 and amd-cloud; within normal part-to-part variation.

<!-- after-results -->
### fp4

**ROCm 7.14 on both sides: our nodes are almost the same as amd-cloud.**

| fp4 TFLOPS (node6100 / node6101) | amd-cloud (7.14) | amd-ubuntu ROCm 7.14 | amd-ubuntu 7.14 / amd-cloud |
|---|---:|---:|---:|
| 1 GPU | 3,976 | 4,079 / 3,990 | 1.03x / 1.00x |
| 8 GPUs, one RVS process (standard run), total | 17,733 | 17,028 / 17,100 | 0.96x / 0.96x |

**Our nodes, ROCm 7.2.4 vs ROCm 7.14** (same nodes; only the ROCm version differs):

| fp4 TFLOPS per GPU (node6100 / node6101) | ROCm 7.2.4 | ROCm 7.14 | 7.14 / 7.2.4 |
|---|---:|---:|---:|
| 1 GPU | 3,187 / 3,175 | 4,079 / 3,990 | **1.28x / 1.26x** |
| 8 GPUs, one RVS process (standard run) | 3,200 / 3,191 | 2,128 / 2,137 | **0.67x / 0.67x** |
| 8 GPUs, 8 separate processes (test 2026-10-05, node6100) | ≈3,200 | ≈4,100 | **1.28x** |

- **1 GPU: ROCm 7.14 has a ≈28% faster fp4 kernel** for MI355X. Other precisions barely change between versions.
- **8 GPUs: the drop on 7.14 comes from the test, not the hardware.** RVS drives all 8 GPUs from one process; on 7.14 that process slows the GPUs down (uneven, 1,800–3,500 each), on amd-cloud as well. 7.2.4 does not.
- **Run as one process per GPU, 7.14 reaches ≈4,100 on all 8 GPUs at once** (32,784 total, 1.28x of 7.2.4). Real workloads run one process per GPU, so they get the 7.14 speed-up.
- Test: `work-rocmval/fp4_multiproc_test.sh`; logs: `logs/node6100/rvs/fp4_multiproc_20261006_002202/`. A recheck and profile is queued: [rvs-fp4-recheck.md](rvs-fp4-recheck.md).
