# amd-ubuntu vs amd-cloud — summary

Written 2026-10-07. Ratio = amd-ubuntu / amd-cloud; for latency (TPOT, TTFT) below 1 is better. Detail per benchmark: the linked reports in this directory.

| | amd-cloud | amd-ubuntu |
|---|---|---|
| Hardware | 1 node, 8 × MI355X | node6100 + node6101, 8 × MI355X each; same amdgpu driver and CPUs |
| OS / ROCm | Ubuntu 22.04.5, ROCm 7.14 | Ubuntu 24.04.5, host ROCm 7.2.4 (+ ROCm 7.14 user space for the apple-to-apple RVS and RCCL runs) |
| Containers | docker | apptainer, **same image digests** |
| Scripts | — | same scripts and analyzers, ported |

**Overall:** with the same software the GPUs and the single-node fabric perform the same (within ±5%). The differences come from the ROCm version (RVS fp4), host-side speed (ATOM serving is faster on amd-ubuntu), or newer recipes run only on amd-ubuntu.

| Benchmark | Result vs amd-cloud | Apple-to-apple? |
|---|---|---|
| [RVS (ROCm Validation Suite)](#1-rvs-rocm-validation-suite-gemm-tflops) | same (0.96–1.05×) on ROCm 7.14; fp4 differs on 7.2.4 | ✅ with ROCm 7.14 |
| [RCCL, one node](#2-rccl-one-node) | same (within 1–3%) at N=8 | ✅ |
| [Primus training](#3-primus-megatron-llama2-7b-training) | same (0.99–1.04×) | ✅ |
| [Megatron-LM GPT-15.6B](#4-megatron-lm-gpt-156b-megatron-ref) | no amd-cloud number | — |
| [ATOM serving](#5-atom-serving-llama-31-70b-qwen3-8b) | amd-ubuntu faster below 256 users (1.07–1.34×) | ✅ |
| [Kimi-K3](#6-kimi-k3-inference) | same (±2%) up to 128 users with ATOM; vLLM recipe is better at low load | ✅ for ATOM; ❌ for the vLLM recipe |
| RDMA, 2-node RCCL | not compared (amd-cloud is one node) | — |

---

## 1. RVS (ROCm Validation Suite), GEMM TFLOPS

[rvs.md](rvs.md)

| | Same | Different |
|---|---|---|
| Setup | test (gst, hipBLASLt), configs, 1 and 8 GPUs | amd-cloud ROCm 7.14; amd-ubuntu host ROCm 7.2.4 and, separately, ROCm 7.14 |
| Result | **ROCm 7.14 on both: 0.96–1.05× for every precision** | fp4 on 7.2.4: 0.80× at N=1, 1.44× at N=8 |

| fp4 TFLOPS | amd-cloud (7.14) | amd-ubuntu 7.14 | amd-ubuntu 7.2.4 |
|---|---:|---:|---:|
| N=1 | 3,976 | 4,079 (1.03×) | 3,187 (0.80×) |
| N=8, total | 17,733 | 17,028 (0.96×) | 25,596 (1.44×) |

The fp4 difference is the ROCm version, not the hardware: 7.14 has a faster fp4 kernel, but with one RVS process driving 8 GPUs it drops (on amd-cloud too). A recheck and profile is queued: [rvs-fp4-recheck.md](rvs-fp4-recheck.md) (written when it finishes).

## 2. RCCL, one node

[rccl.md](rccl.md)

| | Same | Different |
|---|---|---|
| Setup | rccl-tests, sizes, iterations, 2–8 GPUs | ROCm 7.2.4 and 7.14 on amd-ubuntu, 7.14 on amd-cloud |
| Result | **N=8: within 1–3% on every collective**; N=2–4 within 1–2%; the N=5–7 dip appears on both | N=5 alltoallv on 7.2.4 is half of amd-cloud (0.48–0.51×), fixed by 7.14; N=5 gather 0.90–0.94×, scatter 1.09–1.10× |

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
