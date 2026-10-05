# Primus: amd-ubuntu vs amd-cloud

Generated 2026-10-05 18:26 by `report.py`.

- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker ([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).
- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the containers) the same image digests.
- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT, TPOT) below 1 is better.


## Analysis

- **GEMM microbench: amd-ubuntu is 0.97–1.05× amd-cloud** (node6100 1.01–1.05×, node6101 0.97–1.02×); same image digest, so the GPUs match.
- **Megatron llama2-7B**: amd-cloud measured ≈1,070–1,160 compute TF/s/GPU for N=1..8. amd-ubuntu's first run failed (extension build ran out of space in the 64 MB container scratch area); the rerun with a persistent overlay is in progress and the comparison fills in when it finishes.

## Results

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

