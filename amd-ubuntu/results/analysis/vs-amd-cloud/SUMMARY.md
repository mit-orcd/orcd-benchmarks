# amd-ubuntu vs amd-cloud — summary

Hand-written (source `results/analysis/vs-amd-cloud/SUMMARY.md`, copied here by `report.py`); last edited 2026-10-07. Ratio = amd-ubuntu / amd-cloud; for latency (TPOT, TTFT) below 1 is better. Detail per benchmark: the linked reports in this directory.

| | amd-cloud | amd-ubuntu |
|---|---|---|
| Hardware | 1 node, 8 × MI355X | node6100 + node6101, 8 × MI355X each; same amdgpu driver and CPUs |
| OS / ROCm | Ubuntu 22.04.5, ROCm 7.14 | Ubuntu 24.04.5, host ROCm 7.2.4 (+ ROCm 7.14 user space for the apple-to-apple RVS and RCCL runs) |
| Containers | docker | apptainer, **same image digests** |
| Scripts | — | same scripts and analyzers, ported |

**Overall:** with the same ROCm 7.14, our nodes are almost the same as amd-cloud (RVS and RCCL within a few percent). The host ROCm 7.2.4 on our nodes is compared only with ROCm 7.14 on our nodes, never with amd-cloud directly. Other differences come from host-side speed (ATOM serving is faster on amd-ubuntu) or newer recipes run only on amd-ubuntu.

| Benchmark | Result vs amd-cloud | Apple-to-apple? |
|---|---|---|
| [RVS (ROCm Validation Suite)](#1-rvs-rocm-validation-suite-gemm-tflops) | ROCm 7.14: almost the same as amd-cloud (0.96–1.05×). Our 7.2.4 vs 7.14: same except fp4 and fp6/bf6 | ✅ |
| [RCCL, one node](#2-rccl-one-node) | ROCm 7.14: almost the same as amd-cloud (within 1–3% at N=8). Our 7.2.4 vs 7.14: same at N=8; differs at N=5 and across 2 nodes | ✅ |
| [Primus training](#3-primus-megatron-llama2-7b-training) | same (0.99–1.04×) | ✅ |
| [Megatron-LM GPT-15.6B](#4-megatron-lm-gpt-156b-megatron-ref) | no amd-cloud number | — |
| [ATOM serving](#5-atom-serving-llama-31-70b-qwen3-8b) | amd-ubuntu faster below 256 users (1.07–1.34×) | ✅ |
| [Kimi-K3](#6-kimi-k3-inference) | same (±2%) up to 128 users with ATOM; vLLM recipe is better at low load | ✅ for ATOM; ❌ for the vLLM recipe |
| RDMA, 2-node RCCL | not compared (amd-cloud is one node) | — |

---

## 1. RVS (ROCm Validation Suite), GEMM TFLOPS

[rvs.md](rvs.md), [`../ubuntu/rocm.md`](../ubuntu/rocm.md)

**ROCm 7.14 vs amd-cloud (ROCm 7.14): our nodes are almost the same as amd-cloud** — 0.96–1.05× for every precision at 1 and 8 GPUs (same test, same configs).

**Our nodes, ROCm 7.2.4 vs ROCm 7.14** (same nodes, same driver, only the ROCm user space differs):

| | Same | Different |
|---|---|---|
| Precisions | bf16, bf8, fp8, fp16, fp32, fp64: within ±2% | fp4 (table); fp6/bf6 ≈3% lower on 7.14 |

| fp4 TFLOPS per GPU (node6100 / node6101) | ROCm 7.2.4 | ROCm 7.14 | 7.14 / 7.2.4 |
|---|---:|---:|---:|
| 1 GPU | 3,187 / 3,175 | 4,079 / 3,990 | **1.28× / 1.26×** |
| 8 GPUs, one RVS process (standard run) | 3,200 / 3,191 | 2,128 / 2,137 | **0.67× / 0.67×** |
| 8 GPUs, 8 separate processes (test 2026-10-05, node6100) | ≈3,200 | ≈4,100 | **1.28×** |

7.14 has a faster fp4 kernel; with one RVS process driving 8 GPUs it drops (the same drop shows on amd-cloud), with one process per GPU it does not. A recheck and profile is queued: [rvs-fp4-recheck.md](rvs-fp4-recheck.md) (written when it finishes).

## 2. RCCL, one node

[rccl.md](rccl.md), [`../ubuntu/rocm.md`](../ubuntu/rocm.md)

**ROCm 7.14 vs amd-cloud (ROCm 7.14): our nodes are almost the same as amd-cloud** — within 1–3% on every collective at 8 GPUs and within 1–2% at 2–4 GPUs. Only at 5 GPUs do two collectives differ more: gather 0.90–0.94×, scatter 1.09–1.10×. The 5–7 GPU dip appears on both.

**Our nodes, ROCm 7.2.4 vs ROCm 7.14:**

| | Same | Different |
|---|---|---|
| 8 GPUs, one node | every collective within ±2% | — |
| 5 GPUs, one node | most collectives within ±5% | alltoallv 2× faster on 7.14 (≈45 vs ≈22 GB/s); scatter 1.07× ; reduce 0.93–0.95× |
| 2 nodes, 8 GPUs per node | all_reduce, all_gather, reduce_scatter, broadcast within ±2% (≈365–381 GB/s) | alltoall 0.86× on 7.14 (77 vs 90 GB/s); sendrecv 1.12× (32 vs 29 GB/s) |

amd-cloud is one node, so the 2-node rows have no amd-cloud counterpart.

## 3. Primus (Megatron llama2-7B training)

[primus.md](primus.md)

| | Same | Different |
|---|---|---|
| Setup | Primus v26.5 image (same digest), llama2-7B BF16, 1–8 GPUs | apptainer + persistent overlay instead of docker |
| Result | **compute TF/s/GPU 0.99–1.04×** at every N (8 GPUs: 1,170 / 1,135 vs 1,135); GEMM microbench 0.97–1.05× | wall-clock TF/s lower (≈170–270 vs ≈295–375): it includes container start and a one-time extension build, not GPU speed |

## 4. Megatron-LM GPT-15.6B (megatron-ref)

[megatron_ref.md](megatron_ref.md)

| | amd-cloud | amd-ubuntu |
|---|---|---|
| Result | no comparable run | 578.2 (node6100), 572.5 (node6101) TF/s/GPU |

No ratio possible. For context, Dell Cloud MI355X reached 790.4 with the same model; amd-ubuntu is ≈0.73× of that, with fused RoPE turned off (it crashes on gfx950, on amd-cloud as well).

## 5. ATOM serving (Llama-3.1-70B, Qwen3-8B)

[atom.md](atom.md)

| | Same | Different |
|---|---|---|
| Setup | ATOM images (same digests), models, workloads, concurrency 1–256 | host OS and kernel |
| Result | **at 256 users: equal (1.00–1.04×)** | **below 256 users amd-ubuntu is faster** (table) |

| Model | Throughput | TPOT | TTFT |
|---|---:|---:|---:|
| Llama-3.1-70B-FP8 | 1.14–1.34× | 0.74–0.88× | 0.60–0.90× from 8 users; **1.45× slower at 1–4 users** |
| Qwen3-8B-FP8 | 1.07–1.16× | 0.87–0.95× | 0.55–0.80× |

The gain is largest at low load, where host overhead matters, so it points to the host (newer OS/kernel, CPU settings), not the GPUs. Both amd-ubuntu nodes agree within 1%.

## 6. Kimi-K3 inference

[kimi.md](kimi.md)

| | Same | Different |
|---|---|---|
| ATOM (same images) | **throughput within ±2% and TPOT within ±1% up to 128 users**; `max-num-seqs 2048` fails on both (does not fit in memory) | 256–512 users: 1.05× throughput; `max-num-seqs 1024`: 1.15–1.26× throughput |
| AMD vLLM recipe | — | run only on amd-ubuntu (newer vLLM, speculative decoding); compared with amd-cloud's best ATOM, table below |

| Users | 1 | 4 | 8 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| vLLM recipe (amd-ubuntu) / best ATOM (amd-cloud), tok/s | 1.54× | 1.38× | 1.23× | 1.07× | 1.04× | 1.00× |

This last row is recipe vs recipe, not hardware vs hardware. AMD's newer recipe from `amd-kimi-k3-recipe.pdf` is running now; results go to [`../ubuntu/kimi-amd-recipe.md`](../ubuntu/kimi-amd-recipe.md).
