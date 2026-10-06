- **Both nodes healthy and matched.** Every precision is within 3% between node6100 and node6101; fp32/fp64 are identical to 1%.
- **Scaling is linear on host ROCm 7.2.4**: N=8 gives 99–104% of 8 × N=1 for all nine precisions, so there is no node-level power or thermal limit when all 8 GPUs run GEMMs at once.
- Headline N=8 aggregate (node6100): fp8 30.3 PF, bf8 26.9 PF, fp4 25.6 PF, bf16 13.6 PF, fp64 617 TF.

### fp4: ROCm 7.14 vs 7.2.4

| fp4 TFLOPS | ROCm 7.2.4 | ROCm 7.14 |
|---|---:|---:|
| 1 GPU | 3,187 | **4,079** |
| 8 GPUs, one RVS process (standard run), per GPU | **3,200** | 2,128 |
| 8 GPUs, 8 separate processes, per GPU (test 2026-10-05) | 3,200 | **≈4,100** |

- **ROCm 7.14 has a ≈28% faster fp4 kernel** for MI355X; other precisions barely change.
- **The 8-GPU drop on 7.14 comes from the test, not the hardware.** RVS drives all 8 GPUs from one process. On 7.14 that process slows the GPUs down (uneven, 1,800–3,500 each). Run as one process per GPU, all 8 reach ≈4,100 at the same time (32,784 total, 1.28× of 7.2.4).
- Real workloads run one process per GPU, so they get the 7.14 speed-up. amd-cloud (also 7.14, one process) shows the same drop.
- Test: `work-rocmval/fp4_multiproc_test.sh`; logs: `logs/node6100/rvs/fp4_multiproc_20261006_002202/`.
