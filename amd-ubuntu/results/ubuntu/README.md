# amd-ubuntu benchmark results

Generated 2026-10-07 20:37 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 for RVS (ROCm Validation Suite) and rccl-tests, containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Status

| Benchmark | node6100 | node6101 |
|---|---|---|
| RVS gst TFLOPS | ✅ 10-01 23:13 | ✅ 10-01 23:13 |
| RCCL single node | ✅ 10-02 00:13 | ✅ 10-02 00:26 |
| RVS gst TFLOPS, ROCm 7.14 | ✅ 10-02 16:17 | ✅ 10-02 10:29 |
| RCCL single node, ROCm 7.14 | ✅ 10-02 17:12 | ✅ 10-02 11:24 |
| RCCL 2-node, ROCm 7.14 | ✅ 10-05 03:06 | — |
| RDMA per rail | ✅ 10-01 18:32 | ✅ 10-01 18:36 |
| RCCL 2-node | ✅ 10-05 02:58 | — |
| Primus / Megatron | ✅ 10-05 19:37 | ✅ 10-05 19:37 |
| ATOM tiers 1-2 | ✅ 10-05 01:41 | ✅ 10-05 02:49 |
| Kimi-K3 ATOM (Aug-2026 images) | ✅ 10-05 01:41 | — |
| Kimi-K3 vLLM recipe | — | ✅ 10-05 03:19 |

## Reports

- [RVS gst TFLOPS](rvs.md)
- [RCCL collectives, single node (XGMI)](rccl.md)
- [Host ROCm 7.2.4 vs ROCm 7.14 (RVS, RCCL)](rocm.md)
- [Network: RDMA per ionic rail](net.md)
- [RCCL across two nodes](rccl_2node.md)
- [Primus: Megatron-LM llama2-7B and GEMM microbench](primus.md)
- [Megatron-LM GPT-15.6B](megatron_ref.md)
- [ATOM LLM serving (Qwen3-8B, Llama-3.1-70B)](atom.md)
- [Kimi-K3 serving: ATOM and the AMD vLLM recipe](kimi.md)

## Raw

Per-node analyzer reports, CSVs and plots: `results/<node>/`; Kimi-K3 per image set: `results/<node>/kimi-<set>/`; logs: `logs/<node>/`.
