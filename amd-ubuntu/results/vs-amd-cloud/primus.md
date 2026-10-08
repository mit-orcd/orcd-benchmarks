# Primus: amd-ubuntu vs amd-cloud

Generated 2026-10-08 21:47 by `report.py`.

- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker ([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).
- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the containers) the same image digests.
- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT, TPOT = time per output token) below 1 is better.


## Analysis

- **Megatron llama2-7B (same image digest): amd-ubuntu matches amd-cloud, node6100 1.01–1.04×, node6101 0.99–1.00× compute TF/s/GPU** at every N from 1 to 8 (8 GPUs: 1,170.5 and 1,134.8 vs 1,135.2). Both systems show the same small N=5–7 dip.
- **GEMM microbench: 0.97–1.05×** (node6100 1.01–1.05×, node6101 0.97–1.02×).
- Wall-clock TF/s is lower on amd-ubuntu (≈170–270 vs ≈295–375) because that metric includes container start and the first-time apex extension build in the new overlay; it is not a GPU difference.
- Conclusion: for Primus training the GPUs and single-node fabric perform the same as amd-cloud.

## Results

### Megatron-LM llama2-7B BF16 via Primus, compute TF/s/GPU

| N | amd-cloud | node6100 | node6101 | 6100/cloud | 6101/cloud |
|---|---:|---:|---:|---:|---:|
| 1 | 1,160.8 | 1,177.9 | 1,154.6 | 1.01x | 0.99x |
| 2 | 1,122.2 | 1,152.7 | 1,110.0 | 1.03x | 0.99x |
| 3 | 1,127.9 | 1,156.0 | 1,114.0 | 1.02x | 0.99x |
| 4 | 1,139.5 | 1,170.5 | 1,125.8 | 1.03x | 0.99x |
| 5 | 1,088.2 | 1,113.7 | 1,075.7 | 1.02x | 0.99x |
| 6 | 1,076.9 | 1,110.3 | 1,071.8 | 1.03x | 1.00x |
| 7 | 1,069.8 | 1,107.4 | 1,065.9 | 1.04x | 1.00x |
| 8 | 1,135.2 | 1,170.5 | 1,134.8 | 1.03x | 1.00x |

### Primus GEMM microbench, mean TF/s/GPU

| N | amd-cloud | node6100 | node6101 | 6100/cloud | 6101/cloud |
|---|---:|---:|---:|---:|---:|
| 1 | 1,459.2 | 1,480.5 | 1,425.4 | 1.01x | 0.98x |
| 2 | 1,444.5 | 1,479.9 | 1,401.3 | 1.02x | 0.97x |
| 3 | 1,448.7 | 1,499.1 | 1,424.0 | 1.03x | 0.98x |
| 4 | 1,438.5 | 1,499.8 | 1,430.4 | 1.04x | 0.99x |
| 5 | 1,443.6 | 1,505.9 | 1,444.6 | 1.04x | 1.00x |
| 6 | 1,436.4 | 1,511.0 | 1,461.8 | **1.05x** | 1.02x |
| 7 | 1,435.2 | 1,509.8 | 1,463.3 | **1.05x** | 1.02x |
| 8 | 1,443.5 | 1,504.2 | 1,468.0 | 1.04x | 1.02x |

