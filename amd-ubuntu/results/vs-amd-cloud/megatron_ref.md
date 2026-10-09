# Megatron-LM GPT-15.6B: amd-ubuntu vs amd-cloud

Generated 2026-10-09 06:00 by `report.py`.

- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker ([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).
- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the containers) the same image digests.
- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT, TPOT = time per output token) below 1 is better.


## Analysis

- amd-cloud has no comparable GPT-15.6B run with the full flag set, so there is no direct ratio. amd-ubuntu: 578.2 (node6100) and 572.5 (node6101) TF/s/GPU.
- For context, the earlier Dell Cloud MI355X run of the same configuration reached 790.4 TF/s/GPU; amd-ubuntu is ≈0.73× of that, with fused RoPE turned off here because it crashes on gfx950 (the crash also occurred on amd-cloud).

## Results

### megatron-ref GPT-15.6B (Megatron-LM v26.1), TF/s/GPU

| N | amd-cloud | node6100 | node6101 | 6100/cloud | 6101/cloud |
|---|---:|---:|---:|---:|---:|
| 8 | — | 578.2 | 572.5 | — | — |

