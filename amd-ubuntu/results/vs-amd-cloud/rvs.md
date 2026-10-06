# RVS gst TFLOPS: amd-ubuntu vs amd-cloud

Generated 2026-10-06 00:30 by `report.py`.

- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker ([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).
- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the containers) the same image digests.
- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT, TPOT) below 1 is better.


## Analysis

- **Same ROCm 7.14 on both sides (apple-to-apple): amd-ubuntu matches amd-cloud within 0–5%** for every precision at N=1 and N=8 (fp4 0.96–1.03×, all others 1.00–1.05×). The hardware performs the same.
- On host ROCm 7.2.4, fp4 differs: 1.44× amd-cloud at N=8 but 0.80× at N=1. This is a ROCm version effect, not a hardware difference: 7.14's fp4 kernel is ≈28% faster, but when one RVS process drives all 8 GPUs it slows down on 7.14 (amd-cloud included). Run as 8 separate processes, 7.14 reaches ≈4,100 TFLOPS on every GPU (see `../ubuntu/rocm.md`).
- node6100 runs bf16/fp16/fp8 ≈2–3% faster than node6101 and amd-cloud; within normal part-to-part variation.

## Results

Aggregate `gst` TFLOPS. The GEMMs are independent per GPU, so a gap is clocks, power or the ROCm stack, never the interconnect. amd-cloud ran ROCm 7.14; amd-ubuntu ran host ROCm 7.2.4 and, separately, the same ROCm 7.14 user space (`ROCM_STACK=7.14`): the 7.14 columns are the apple-to-apple comparison.

### N = 8

| precision | amd-cloud (7.14) | 6100 ROCm 7.2.4 | 6101 ROCm 7.2.4 | 6100 ROCm 7.14 | 6101 ROCm 7.14 | 6100 7.2.4/cloud | 6101 7.2.4/cloud | 6100 7.14/cloud | 6101 7.14/cloud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 13,113.8 | 13,556.5 | 13,191.2 | 13,558.4 | 13,200.5 | 1.03x | 1.01x | 1.03x | 1.01x |
| bf6 | 9,908.8 | 10,264.9 | 10,268.3 | 9,914.4 | 9,915.2 | 1.04x | 1.04x | 1.00x | 1.00x |
| bf8 | 26,423.6 | 26,948.0 | 26,861.6 | 27,282.0 | 26,600.0 | 1.02x | 1.02x | 1.03x | 1.01x |
| fp16 | 12,301.5 | 12,703.1 | 12,442.3 | 12,764.0 | 12,484.6 | 1.03x | 1.01x | 1.04x | 1.01x |
| fp32 | 1,228.6 | 1,231.6 | 1,229.5 | 1,230.9 | 1,228.8 | 1.00x | 1.00x | 1.00x | 1.00x |
| fp4 | 17,733.2 | 25,596.3 | 25,530.0 | 17,027.5 | 17,099.5 | **1.44x** | **1.44x** | 0.96x | 0.96x |
| fp6 | 9,910.2 | 10,266.3 | 10,267.8 | 9,925.1 | 9,913.7 | 1.04x | 1.04x | 1.00x | 1.00x |
| fp64 | 614.9 | 616.6 | 615.6 | 616.1 | 615.0 | 1.00x | 1.00x | 1.00x | 1.00x |
| fp8 | 28,816.5 | 30,275.3 | 29,690.4 | 30,278.5 | 29,312.9 | **1.05x** | 1.03x | **1.05x** | 1.02x |

### N = 1

| precision | amd-cloud (7.14) | 6100 ROCm 7.2.4 | 6101 ROCm 7.2.4 | 6100 ROCm 7.14 | 6101 ROCm 7.14 | 6100 7.2.4/cloud | 6101 7.2.4/cloud | 6100 7.14/cloud | 6101 7.14/cloud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 1,628.0 | 1,672.2 | 1,617.2 | 1,670.7 | 1,631.1 | 1.03x | 0.99x | 1.03x | 1.00x |
| bf6 | 1,238.6 | 1,282.2 | 1,285.5 | 1,240.9 | 1,240.9 | 1.04x | 1.04x | 1.00x | 1.00x |
| bf8 | 3,220.4 | 3,393.6 | 3,321.8 | 3,377.9 | 3,342.0 | **1.05x** | 1.03x | 1.05x | 1.04x |
| fp16 | 1,521.8 | 1,586.6 | 1,558.5 | 1,570.5 | 1,519.6 | 1.04x | 1.02x | 1.03x | 1.00x |
| fp32 | 152.8 | 152.4 | 154.3 | 152.6 | 154.1 | 1.00x | 1.01x | 1.00x | 1.01x |
| fp4 | 3,975.7 | 3,187.3 | 3,175.0 | 4,079.3 | 3,989.8 | **0.80x** | **0.80x** | 1.03x | 1.00x |
| fp6 | 1,238.0 | 1,282.0 | 1,286.5 | 1,240.5 | 1,240.9 | 1.04x | 1.04x | 1.00x | 1.00x |
| fp64 | 76.6 | 76.7 | 77.3 | 76.5 | 77.2 | 1.00x | 1.01x | 1.00x | 1.01x |
| fp8 | 3,564.1 | 3,683.5 | 3,558.3 | 3,752.6 | 3,628.8 | 1.03x | 1.00x | **1.05x** | 1.02x |

