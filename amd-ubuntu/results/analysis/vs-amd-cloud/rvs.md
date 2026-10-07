- **Same ROCm 7.14 on both sides (apple-to-apple): amd-ubuntu matches amd-cloud within 0–5%** for every precision at N=1 and N=8 (fp4 0.96–1.03×, all others 1.00–1.05×). The hardware performs the same.
- node6100 runs bf16/fp16/fp8 ≈2–3% faster than node6101 and amd-cloud; within normal part-to-part variation.

### fp4: why amd-ubuntu is slower at N=1 but faster at N=8

The fp4 gap comes from the ROCm version, not the hardware. amd-cloud ran ROCm 7.14; the 7.2.4 columns are amd-ubuntu's host ROCm.

| fp4 TFLOPS (node6100 / node6101) | amd-cloud (7.14) | amd-ubuntu ROCm 7.14 | amd-ubuntu ROCm 7.2.4 |
|---|---:|---:|---:|
| N=1 | 3,976 | 4,079 / 3,990 (**1.03x / 1.00x**) | 3,187 / 3,175 (0.80x / 0.80x) |
| N=8, one RVS process (standard run), total | 17,733 | 17,028 / 17,100 (**0.96x / 0.96x**) | 25,596 / 25,530 (1.44x / 1.44x) |
| N=8, one RVS process, per GPU | 2,217 | 2,128 / 2,137 | 3,200 / 3,191 |
| N=8, 8 separate processes, per GPU (test 2026-10-05, node6100) | — | **≈4,100** | ≈3,200 |

Ratios in brackets are to amd-cloud.

- **N=1: ROCm 7.14 has a ≈28% faster fp4 kernel** for MI355X, so amd-cloud (7.14) beats amd-ubuntu on 7.2.4 (0.80x). Other precisions barely change between versions.
- **N=8: the drop on 7.14 comes from the test, not the hardware.** RVS drives all 8 GPUs from one process; on 7.14 that process slows the GPUs down (uneven, 1,800–3,500 each), on amd-cloud as well. 7.2.4 does not, so amd-ubuntu on 7.2.4 looks 1.44x faster.
- **Run as one process per GPU, 7.14 reaches ≈4,100 on all 8 GPUs at once** (32,784 total, 1.28x of 7.2.4). Real workloads run one process per GPU, so they get the 7.14 speed-up.
- **Same ROCm 7.14 on both sides, amd-ubuntu matches amd-cloud (0.96–1.03x).**
- Test: `work-rocmval/fp4_multiproc_test.sh`; logs: `logs/node6100/rvs/fp4_multiproc_20261006_002202/`.
