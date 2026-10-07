# RVS (ROCm Validation Suite) gst TFLOPS: amd-ubuntu vs amd-cloud

Generated 2026-10-07 20:01 by `report.py`.

- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker ([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).
- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the containers) the same image digests.
- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT, TPOT) below 1 is better.


## Analysis

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

## Results

Aggregate `gst` TFLOPS. The GEMMs are independent per GPU, so a gap is clocks, power or the ROCm stack, never the interconnect. amd-cloud ran ROCm 7.14; amd-ubuntu ran host ROCm 7.2.4 and, separately, the same ROCm 7.14 user space (`ROCM_STACK=7.14`): the 7.14 columns are the apple-to-apple comparison.

### N = 8

| precision | amd-cloud (7.14) | 6100 ROCm 7.14 | 6101 ROCm 7.14 | 6100 ROCm 7.2.4 | 6101 ROCm 7.2.4 | 6100 7.14/cloud | 6101 7.14/cloud | 6100 7.2.4/cloud | 6101 7.2.4/cloud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 13,113.8 | 13,558.4 | 13,200.5 | 13,556.5 | 13,191.2 | 1.03x | 1.01x | 1.03x | 1.01x |
| bf6 | 9,908.8 | 9,914.4 | 9,915.2 | 10,264.9 | 10,268.3 | 1.00x | 1.00x | 1.04x | 1.04x |
| bf8 | 26,423.6 | 27,282.0 | 26,600.0 | 26,948.0 | 26,861.6 | 1.03x | 1.01x | 1.02x | 1.02x |
| fp16 | 12,301.5 | 12,764.0 | 12,484.6 | 12,703.1 | 12,442.3 | 1.04x | 1.01x | 1.03x | 1.01x |
| fp32 | 1,228.6 | 1,230.9 | 1,228.8 | 1,231.6 | 1,229.5 | 1.00x | 1.00x | 1.00x | 1.00x |
| fp4 | 17,733.2 | 17,027.5 | 17,099.5 | 25,596.3 | 25,530.0 | 0.96x | 0.96x | **1.44x** | **1.44x** |
| fp6 | 9,910.2 | 9,925.1 | 9,913.7 | 10,266.3 | 10,267.8 | 1.00x | 1.00x | 1.04x | 1.04x |
| fp64 | 614.9 | 616.1 | 615.0 | 616.6 | 615.6 | 1.00x | 1.00x | 1.00x | 1.00x |
| fp8 | 28,816.5 | 30,278.5 | 29,312.9 | 30,275.3 | 29,690.4 | **1.05x** | 1.02x | **1.05x** | 1.03x |

### N = 1

| precision | amd-cloud (7.14) | 6100 ROCm 7.14 | 6101 ROCm 7.14 | 6100 ROCm 7.2.4 | 6101 ROCm 7.2.4 | 6100 7.14/cloud | 6101 7.14/cloud | 6100 7.2.4/cloud | 6101 7.2.4/cloud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 1,628.0 | 1,670.7 | 1,631.1 | 1,672.2 | 1,617.2 | 1.03x | 1.00x | 1.03x | 0.99x |
| bf6 | 1,238.6 | 1,240.9 | 1,240.9 | 1,282.2 | 1,285.5 | 1.00x | 1.00x | 1.04x | 1.04x |
| bf8 | 3,220.4 | 3,377.9 | 3,342.0 | 3,393.6 | 3,321.8 | 1.05x | 1.04x | **1.05x** | 1.03x |
| fp16 | 1,521.8 | 1,570.5 | 1,519.6 | 1,586.6 | 1,558.5 | 1.03x | 1.00x | 1.04x | 1.02x |
| fp32 | 152.8 | 152.6 | 154.1 | 152.4 | 154.3 | 1.00x | 1.01x | 1.00x | 1.01x |
| fp4 | 3,975.7 | 4,079.3 | 3,989.8 | 3,187.3 | 3,175.0 | 1.03x | 1.00x | **0.80x** | **0.80x** |
| fp6 | 1,238.0 | 1,240.5 | 1,240.9 | 1,282.0 | 1,286.5 | 1.00x | 1.00x | 1.04x | 1.04x |
| fp64 | 76.6 | 76.5 | 77.2 | 76.7 | 77.3 | 1.00x | 1.01x | 1.00x | 1.01x |
| fp8 | 3,564.1 | 3,752.6 | 3,628.8 | 3,683.5 | 3,558.3 | **1.05x** | 1.02x | 1.03x | 1.00x |

