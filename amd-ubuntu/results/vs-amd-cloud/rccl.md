# RCCL single node: amd-ubuntu vs amd-cloud

Generated 2026-10-06 00:30 by `report.py`.

- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker ([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).
- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the containers) the same image digests.
- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT, TPOT) below 1 is better.


## Analysis

- **N=8: amd-ubuntu equals amd-cloud within 1–3% on every collective**, on both 7.2.4 and 7.14. The XGMI fabric performs the same.
- N=2..4: identical within 1–2%.
- N=5..7 dip is present on both systems. With the same 7.14 stack the N=5 numbers match within 1% except gather (0.90–0.94×) and scatter (1.09–1.10×). On host 7.2.4, alltoallv at N=5 is half the amd-cloud value (0.48–0.51×); 7.14 fixes that.

## Results

busbw (GB/s) at the largest message size. amd-cloud ran ROCm/RCCL 7.14; the amd-ubuntu 7.14 columns use the same ROCm 7.14 user space (apple-to-apple), the 7.2.4 columns the host ROCm.

### N = 8 (full XGMI mesh)

| collective | amd-cloud (7.14) | 6100 ROCm 7.2.4 | 6101 ROCm 7.2.4 | 6100 ROCm 7.14 | 6101 ROCm 7.14 | 6100 7.2.4/cloud | 6101 7.2.4/cloud | 6100 7.14/cloud | 6101 7.14/cloud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 388.1 | 388.4 | 387.7 | 389.0 | 389.7 | 1.00x | 1.00x | 1.00x | 1.00x |
| all_reduce | 396.3 | 394.4 | 393.1 | 395.8 | 398.0 | 1.00x | 0.99x | 1.00x | 1.00x |
| alltoall | 346.4 | 343.1 | 342.6 | 347.6 | 347.5 | 0.99x | 0.99x | 1.00x | 1.00x |
| alltoallv | 212.2 | 216.8 | 217.0 | 218.1 | 220.4 | 1.02x | 1.02x | 1.03x | 1.04x |
| broadcast | 389.6 | 386.7 | 387.1 | 389.1 | 389.7 | 0.99x | 0.99x | 1.00x | 1.00x |
| gather | 426.0 | 425.4 | 424.7 | 424.8 | 422.8 | 1.00x | 1.00x | 1.00x | 0.99x |
| reduce | 330.4 | 327.4 | 329.2 | 322.0 | 328.6 | 0.99x | 1.00x | 0.97x | 0.99x |
| reduce_scatter | 387.3 | 386.3 | 386.6 | 388.4 | 388.9 | 1.00x | 1.00x | 1.00x | 1.00x |
| scatter | 396.6 | 397.1 | 397.1 | 395.6 | 396.1 | 1.00x | 1.00x | 1.00x | 1.00x |
| sendrecv | 60.2 | 60.7 | 60.2 | 60.3 | 60.5 | 1.01x | 1.00x | 1.00x | 1.00x |

### N = 5 (inside the N=5..7 dip)

| collective | amd-cloud (7.14) | 6100 ROCm 7.2.4 | 6101 ROCm 7.2.4 | 6100 ROCm 7.14 | 6101 ROCm 7.14 | 6100 7.2.4/cloud | 6101 7.2.4/cloud | 6100 7.14/cloud | 6101 7.14/cloud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 45.8 | 46.0 | 45.9 | 45.9 | 46.0 | 1.00x | 1.00x | 1.00x | 1.00x |
| all_reduce | 48.0 | 47.1 | 47.2 | 48.1 | 48.2 | 0.98x | 0.98x | 1.00x | 1.00x |
| alltoall | 61.4 | 61.0 | 60.8 | 61.5 | 61.2 | 0.99x | 0.99x | 1.00x | 1.00x |
| alltoallv | 45.4 | 22.0 | 23.0 | 45.4 | 45.2 | **0.48x** | **0.51x** | 1.00x | 1.00x |
| broadcast | 39.6 | 41.9 | 41.7 | 39.9 | 40.0 | **1.06x** | **1.05x** | 1.01x | 1.01x |
| gather | 78.3 | 70.4 | 70.2 | 73.2 | 70.7 | **0.90x** | **0.90x** | **0.94x** | **0.90x** |
| reduce | 45.7 | 48.6 | 49.5 | 46.4 | 45.8 | **1.06x** | **1.08x** | 1.01x | 1.00x |
| reduce_scatter | 46.3 | 47.6 | 47.9 | 46.4 | 46.7 | 1.03x | 1.03x | 1.00x | 1.01x |
| scatter | 69.4 | 71.1 | 69.8 | 76.3 | 75.7 | 1.02x | 1.01x | **1.10x** | **1.09x** |
| sendrecv | 60.3 | 61.0 | 60.5 | 60.5 | 60.1 | 1.01x | 1.00x | 1.00x | 1.00x |

### N = 2 (one link)

| collective | amd-cloud (7.14) | 6100 ROCm 7.2.4 | 6101 ROCm 7.2.4 | 6100 ROCm 7.14 | 6101 ROCm 7.14 | 6100 7.2.4/cloud | 6101 7.2.4/cloud | 6100 7.14/cloud | 6101 7.14/cloud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 59.6 | 59.6 | 59.8 | 59.3 | 59.5 | 1.00x | 1.00x | 1.00x | 1.00x |
| all_reduce | 59.6 | 59.6 | 59.4 | 59.6 | 59.9 | 1.00x | 1.00x | 1.00x | 1.00x |
| alltoall | 58.2 | 58.5 | 58.0 | 58.7 | 59.1 | 1.01x | 1.00x | 1.01x | 1.02x |
| alltoallv | 58.3 | 58.3 | 58.1 | 58.7 | 59.1 | 1.00x | 1.00x | 1.01x | 1.01x |
| broadcast | 62.0 | 62.0 | 61.8 | 62.0 | 62.7 | 1.00x | 1.00x | 1.00x | 1.01x |
| gather | 61.6 | 61.4 | 61.5 | 61.8 | 61.8 | 1.00x | 1.00x | 1.00x | 1.00x |
| reduce | 62.1 | 61.7 | 61.8 | 61.5 | 61.8 | 0.99x | 1.00x | 0.99x | 1.00x |
| reduce_scatter | 57.8 | 55.8 | 55.8 | 56.9 | 57.6 | 0.97x | 0.97x | 0.99x | 1.00x |
| scatter | 61.4 | 61.6 | 61.3 | 61.9 | 62.1 | 1.00x | 1.00x | 1.01x | 1.01x |
| sendrecv | 58.4 | 58.6 | 58.1 | 58.5 | 58.4 | 1.00x | 0.99x | 1.00x | 1.00x |

### Config sweep, N = 8

| collective | config | amd-cloud (7.14) | 6100 ROCm 7.2.4 | 6101 ROCm 7.2.4 | 6100 ROCm 7.14 | 6101 ROCm 7.14 | 6100 7.2.4/cloud | 6101 7.2.4/cloud | 6100 7.14/cloud | 6101 7.14/cloud |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | no_mscll | 389.1 | 386.8 | 387.0 | 389.7 | 390.6 | 0.99x | 0.99x | 1.00x | 1.00x |
| all_gather | proto_simple | 389.6 | 388.8 | 387.3 | 390.1 | 389.2 | 1.00x | 0.99x | 1.00x | 1.00x |
| all_gather | ring | 388.9 | 388.2 | 386.1 | 388.8 | 390.3 | 1.00x | 0.99x | 1.00x | 1.00x |
| all_gather | tree | 388.5 | 387.6 | 386.6 | 389.3 | 387.9 | 1.00x | 0.99x | 1.00x | 1.00x |
| all_reduce | no_mscll | 396.8 | 394.4 | 393.0 | 396.9 | 396.0 | 0.99x | 0.99x | 1.00x | 1.00x |
| all_reduce | proto_simple | 397.0 | 393.7 | 392.7 | 397.2 | 397.4 | 0.99x | 0.99x | 1.00x | 1.00x |
| all_reduce | ring | 396.3 | 394.6 | 392.8 | 396.3 | 394.4 | 1.00x | 0.99x | 1.00x | 1.00x |
| all_reduce | tree | 167.0 | 171.0 | 170.6 | 170.8 | 172.4 | 1.02x | 1.02x | 1.02x | 1.03x |
