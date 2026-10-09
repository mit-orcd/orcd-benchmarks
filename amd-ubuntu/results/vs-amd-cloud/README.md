# amd-ubuntu vs amd-cloud

Generated 2026-10-09 06:00 by `report.py`.

- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker ([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).
- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the containers) the same image digests.
- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT = time to first token, TPOT = time per output token) below 1 is better.


## Status

| Benchmark | node6100 | node6101 |
|---|---|---|
| RVS (ROCm Validation Suite) gst TFLOPS | ✅ 10-01 23:13 | ✅ 10-01 23:13 |
| RCCL single node | ✅ 10-02 00:13 | ✅ 10-02 00:26 |
| RVS gst TFLOPS, ROCm 7.14 | ✅ 10-02 16:17 | ✅ 10-02 10:29 |
| RCCL single node, ROCm 7.14 | ✅ 10-02 17:12 | ✅ 10-02 11:24 |
| RCCL 2-node, ROCm 7.14 | ✅ 10-05 03:06 | — |
| RDMA per rail | ✅ 10-01 18:32 | ✅ 10-01 18:36 |
| RCCL 2-node | ✅ 10-05 02:58 | — |
| Primus / Megatron | ✅ 10-05 19:37 | ✅ 10-05 19:37 |
| ATOM tiers 1-2 | ✅ 10-05 01:41 | ✅ 10-05 02:49 |
| Kimi-K3 ATOM (Aug-2026 images) | ✅ 10-05 01:41 | — |
| Kimi-K3 vLLM recipe 1 | — | ✅ 10-05 03:19 |

## Reports

- [**Summary of all results (what vs what, per benchmark)**](../SUMMARY.md)
- [RVS gst TFLOPS: amd-ubuntu vs amd-cloud](rvs.md)
- [RCCL single node: amd-ubuntu vs amd-cloud](rccl.md)
- [Primus: amd-ubuntu vs amd-cloud](primus.md)
- [Megatron-LM GPT-15.6B: amd-ubuntu vs amd-cloud](megatron_ref.md)
- [ATOM serving: amd-ubuntu vs amd-cloud](atom.md)
- [Kimi-K3: amd-ubuntu vs amd-cloud](kimi.md)

## Not compared

RDMA per rail and 2-node RCCL: amd-cloud was a single node. See [`../ubuntu/net.md`](../ubuntu/net.md), [`../ubuntu/rccl_2node.md`](../ubuntu/rccl_2node.md).
