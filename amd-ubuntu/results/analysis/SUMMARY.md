# amd-ubuntu results — summary

Hand-written (source `results/analysis/SUMMARY.md`, copied to `results/SUMMARY.md` by `report.py`); last edited 2026-10-08. Covers every benchmark on our two MI355X nodes. Each section starts with what is compared with what. Ratios: for latency (TPOT, TTFT) below 1 is better.

Detail: [`ubuntu/`](ubuntu/README.md) (our nodes only), [`vs-amd-cloud/`](vs-amd-cloud/README.md) (vs amd-cloud), [`mi355x-vs-b200.md`](mi355x-vs-b200.md) (vs NVIDIA B200), [`kimi-summary.md`](kimi-summary.md) (all Kimi-K3 runs on one page).

| | our nodes (amd-ubuntu) | amd-cloud |
|---|---|---|
| Hardware | node6100 + node6101, 8 × MI355X each; 8 × 400G AMD Pollara RoCEv2 rails per node | 1 node, 8 × MI355X; same amdgpu driver and CPUs |
| OS / ROCm | Ubuntu 24.04.5, host ROCm 7.2.4, plus a ROCm 7.14 user space for RVS and RCCL | Ubuntu 22.04.5, ROCm 7.14 |
| Containers | apptainer, **same image digests** as amd-cloud | docker |

ROCm 7.14 is newer than ROCm 7.2.4 (our host version), so we compare our 7.14 only with amd-cloud's 7.14 and our 7.2.4 only with our 7.14.

**Rules used throughout:** our nodes are compared with amd-cloud only on the same ROCm 7.14; our ROCm 7.2.4 is compared only with ROCm 7.14 on our nodes, never with amd-cloud directly.

| # | Benchmark | Compared | Result |
|---|---|---|---|
| 2 | [RVS (ROCm Validation Suite) GEMM](#2-rvs-rocm-validation-suite-gemm-tflops) | ours 7.14 vs amd-cloud 7.14; ours 7.2.4 vs ours 7.14 | almost the same as amd-cloud; 7.2.4 = 7.14 except fp4 |
| 3 | [RCCL, one node](#3-rccl-one-node) | ours 7.14 vs amd-cloud 7.14; ours 7.2.4 vs ours 7.14 | almost the same as amd-cloud; 7.2.4 = 7.14 at 8 GPUs |
| 4 | [Network and RCCL, two nodes](#4-network-and-rccl-two-nodes-our-nodes-only) | our nodes only: vs line rate; 7.2.4 vs 7.14 | 98% of line rate per rail; collectives 92–95% of 400 GB/s; sendrecv 94% with one RCCL setting |
| 5 | [Primus training](#5-primus-megatron-llama2-7b-training) | ours vs amd-cloud, same image | the same (0.99–1.04×) |
| 6 | [Megatron-LM GPT-15.6B](#6-megatron-lm-gpt-156b-megatron-ref) | our nodes only (no amd-cloud run) | 578 / 573 TF/s/GPU |
| 7 | [ATOM serving](#7-atom-serving-llama-31-70b-qwen3-8b) | ours vs amd-cloud, same images | ours faster below 256 users (1.07–1.34×), equal at 256 |
| 8 | [Kimi-K3](#8-kimi-k3-inference) | ours vs amd-cloud (ATOM, same images); vLLM recipe 1 vs ATOM; vLLM recipe 2 vs its published numbers | the same up to 128 users; recipe faster at low load; vLLM recipe 2 matches AMD's numbers from 8 users |

---

## 1. Quick summary

**Our nodes vs amd-cloud (same ROCm 7.14, same container images):** RVS GEMM, single-node RCCL, Primus training and Kimi-K3 with ATOM up to 128 users are almost the same (within a few percent).

**Different** (all on our nodes vs amd-cloud):
- ATOM serving of Llama-3.1-70B and Qwen3-8B is 7–34% faster on our nodes than on amd-cloud below 256 users (host side, not the GPUs).
- Kimi-K3 with ATOM is 5–26% faster on our nodes than on amd-cloud from 256 users up.
- At 5 GPUs, RCCL gather is 6–10% slower and scatter 9–10% faster on our nodes than on amd-cloud.

**Our nodes, ROCm 7.2.4 vs 7.14:**

**Same:** RVS for all precisions except fp4 (within ±3%); RCCL at 8 GPUs in one node, and the ring collectives across two nodes (within ±2%).

**Different** (all ROCm 7.14 vs 7.2.4 on our nodes):
- fp4 on 1 GPU: 7.14 is 1.28× faster than 7.2.4.
- fp4 on 8 GPUs driven by one RVS process: 7.14 is 33% slower than 7.2.4 (0.67×). Profiled: the GPUs wait for the one process, the kernel itself is fast.
- RCCL alltoallv at 5 GPUs: 7.14 is 2× faster than 7.2.4.
- Across two nodes, alltoall: 7.14 is 14% slower than 7.2.4.
- Across two nodes, sendrecv: 7.14 is 12% faster than 7.2.4.

**Our nodes only, fixes found:**
- 2-node sendrecv: `NCCL_NCHANNELS_PER_NET_PEER=4` raises it from 57% to 94% of one rail (28.5 → 46.9 GB/s).

**Kimi-K3, vLLM recipe 2 on our nodes:** with AMD's own workload (128K/1K), running that same recipe, our nodes match AMD's published numbers at 8–16 users and are 7–10% faster at 32–128, but 18–44% slower at 1–4 users. On the 1K/1K workload it gives 54 tok/s at 1 user up to 1,735 tok/s at 128 users.

## 2. RVS (ROCm Validation Suite), GEMM TFLOPS

**Compared: (a) our nodes on ROCm 7.14 vs amd-cloud on ROCm 7.14; (b) our nodes, ROCm 7.2.4 vs ROCm 7.14.** Same test (gst, hipBLASLt), same configs, 1 and 8 GPUs. Detail: [vs-amd-cloud/rvs.md](vs-amd-cloud/rvs.md), [ubuntu/rocm.md](ubuntu/rocm.md).

**(a) Our nodes are almost the same as amd-cloud:** 0.96–1.05× for every precision at 1 and 8 GPUs.

**(b) ROCm 7.2.4 vs 7.14 on our nodes** (same nodes and driver, only the ROCm user space differs):

| | Same | Different |
|---|---|---|
| Precisions | all except fp4: within ±3% (bf16, bf8, fp8, fp16, fp32, fp64 ±2%; fp6/bf6 −3%) | fp4 (table) |

| fp4 TFLOPS per GPU (node6100 / node6101) | ROCm 7.2.4 | ROCm 7.14 | 7.14 / 7.2.4 |
|---|---:|---:|---:|
| 1 GPU | 3,187 / 3,175 | 4,079 / 3,990 | **1.28× / 1.26×** |
| 8 GPUs, one RVS process (standard run) | 3,200 / 3,191 | 2,128 / 2,137 | **0.67× / 0.67×** |
| 8 GPUs, 8 separate processes (test 2026-10-05, node6100) | ≈3,200 | ≈4,100 | **1.28×** |

7.14 has a faster fp4 kernel; with one RVS process driving 8 GPUs it drops (amd-cloud shows the same drop), with one process per GPU it does not. **Recheck and profile (2026-10-08) confirm it:** 3,984 per GPU on 1 GPU, 2,284 with 8 GPUs in one process, 4,029 with one process per GPU. The FP4 kernel runs at the same speed in all cases, but with one process each GPU is busy only 30–63% of the time (1 GPU: 91%), so **the cause is the single host process, not the GPUs**. Detail: [vs-amd-cloud/rvs-fp4-recheck.md](vs-amd-cloud/rvs-fp4-recheck.md).

Headline on our nodes (ROCm 7.2.4, 8 GPUs, node6100): fp8 30.3 PF, bf8 26.9 PF, fp4 25.6 PF, bf16 13.6 PF, fp64 617 TF; scaling 1 → 8 GPUs is linear (99–104%).

## 3. RCCL, one node

**Compared: (a) our nodes on ROCm 7.14 vs amd-cloud on ROCm 7.14; (b) our nodes, ROCm 7.2.4 vs ROCm 7.14.** rccl-tests, 2–8 GPUs inside one node (XGMI). Detail: [vs-amd-cloud/rccl.md](vs-amd-cloud/rccl.md), [ubuntu/rccl.md](ubuntu/rccl.md), [ubuntu/rocm.md](ubuntu/rocm.md).

**(a) Our nodes are almost the same as amd-cloud:** within 1–3% on every collective at 8 GPUs, 1–2% at 2–4 GPUs. Only at 5 GPUs do two collectives differ more (gather 0.90–0.94×, scatter 1.09–1.10×). The 5–7 GPU dip appears on both.

**(b) ROCm 7.2.4 vs 7.14 on our nodes:**

| | Same | Different |
|---|---|---|
| 8 GPUs | every collective within ±2% (all_reduce ≈394–398 GB/s) | — |
| 5 GPUs | most collectives within ±5% | alltoallv 2× faster on 7.14 (≈45 vs ≈22 GB/s); scatter 1.07×; reduce 0.93–0.95× |

Use 1, 2, 4 or 8 GPUs per node for collective-heavy work: with 5–7 GPUs busbw drops to ≈47 GB/s (a known partial-mesh effect, same on amd-cloud).

## 4. Network and RCCL, two nodes (our nodes only)

**Compared: our nodes only — measured bandwidth vs the network's line rate, and ROCm 7.2.4 vs ROCm 7.14.** amd-cloud was a single node, so there is nothing to compare there. node6100 + node6101, 8 × 400G Pollara RoCEv2 rails per node = 400 GB/s per node. Detail: [ubuntu/net.md](ubuntu/net.md), [ubuntu/rccl_2node.md](ubuntu/rccl_2node.md), [ubuntu/rocm.md](ubuntu/rocm.md).

**RDMA per rail (`ib_write_bw`):** all 16 rails run at ≈392 Gb/s = **98% of line rate**, alone and all 8 at once (≈391–392 GB/s per node). Each process must be bound to its NIC's CPU socket (NUMA node); without that, 8 rails together reach only 52%.

**RCCL across the two nodes (ROCm 7.2.4), busbw at the largest message:**

| Collective | GPUs | busbw GB/s | ceiling GB/s | % of ceiling |
|---|---|---:|---:|---:|
| all_reduce | 16 (2 × 8) | 380.9 | 400 | 95% |
| reduce_scatter | 16 (2 × 8) | 378.5 | 400 | 95% |
| all_gather | 16 (2 × 8) | 376.7 | 400 | 94% |
| broadcast | 16 (2 × 8) | 369.9 | 400 | 92% |
| alltoall | 16 (2 × 8) | 89.6 | 400 (upper bound) | 22% |
| sendrecv | 16 (2 × 8) | 28.6 | 50 (one rail per GPU pair) | 57% |
| all_reduce | 2 (2 × 1) / 4 (2 × 2) / 8 (2 × 4) | 48.7 / 48.7 / 159.8 | 50 / 100 / 200 | 97% / 49% / 80% |

- With 8 GPUs per node the ring collectives reach 92–95% of the network, close to the single-node XGMI rate (≈390 GB/s): the network is not a bottleneck for data-parallel training across the two nodes.
- alltoall is well below its ceiling, so traffic like MoE expert-parallel across nodes will be limited by it.
- **sendrecv is fixed by one RCCL setting** (tested 2026-10-08, ROCm 7.2.4, [ubuntu/sendrecv-check.md](ubuntu/sendrecv-check.md)): `NCCL_NCHANNELS_PER_NET_PEER=4` (or 8) raises it from 28.5 to **46.9 GB/s = 94% of one rail**. The default gives each network peer too few channels; more queue pairs, a larger chunk size or PXN did not help.

**ROCm 7.2.4 vs 7.14 (8 GPUs per node):**

| | Same | Different |
|---|---|---|
| Collectives | all_reduce, all_gather, reduce_scatter, broadcast within ±2% (≈365–381 GB/s) | alltoall 0.86× on 7.14 (77 vs 90 GB/s); sendrecv 1.12× (32 vs 29 GB/s) |

## 5. Primus (Megatron llama2-7B training)

**Compared: our nodes vs amd-cloud, same Primus v26.5 image (same digest).** llama2-7B BF16, 1–8 GPUs. Detail: [vs-amd-cloud/primus.md](vs-amd-cloud/primus.md), [ubuntu/primus.md](ubuntu/primus.md).

| | Same | Different |
|---|---|---|
| Setup | image, model, batch sizes, 1–8 GPUs | apptainer + persistent overlay instead of docker |
| Result | **compute TF/s/GPU 0.99–1.04×** at every N (8 GPUs: 1,170 / 1,135 vs 1,135); GEMM microbench 0.97–1.05× | wall-clock TF/s lower (≈170–270 vs ≈295–375): it includes container start and a one-time extension build, not GPU speed |

## 6. Megatron-LM GPT-15.6B (megatron-ref)

**Compared: our nodes only** — amd-cloud has no comparable run. 8 GPUs, Megatron-LM v26.1 image. Detail: [ubuntu/megatron_ref.md](ubuntu/megatron_ref.md).

| | node6100 | node6101 |
|---|---:|---:|
| TF/s/GPU | 578.2 | 572.5 |

For context only: Dell Cloud MI355X reached 790.4 with the same model; ours is ≈0.73× of that, with fused RoPE turned off (it crashes on gfx950).

## 7. ATOM serving (Llama-3.1-70B, Qwen3-8B)

**Compared: our nodes vs amd-cloud, same ATOM images (same digests), same models and workloads, 1–256 users.** Ratio = ours / amd-cloud. Detail: [vs-amd-cloud/atom.md](vs-amd-cloud/atom.md), [ubuntu/atom.md](ubuntu/atom.md).

| Model | Throughput | TPOT | TTFT |
|---|---:|---:|---:|
| Llama-3.1-70B-FP8 | 1.14–1.34× | 0.74–0.88× | 0.60–0.90× from 8 users; **1.45× slower at 1–4 users** |
| Qwen3-8B-FP8 | 1.07–1.16× | 0.87–0.95× | 0.55–0.80× |

- At 256 users both are equal (1.00–1.04×), where the GPUs are the limit.
- The gain is largest at low load, where host overhead matters, so it points to the host (newer OS/kernel, CPU settings), not the GPUs. Both of our nodes agree within 1%.

## 8. Kimi-K3 inference

**Compared: (a) our nodes vs amd-cloud with ATOM, same images; (b) on our nodes, AMD's vLLM recipe 1 vs the best ATOM result; (c) vLLM recipe 2: our nodes vs AMD's published numbers, and its results on our workload.** ISL/OSL 1024/1024 (plus 128K/1K in (c)), 8 GPUs (TP8). Detail: [vs-amd-cloud/kimi.md](vs-amd-cloud/kimi.md), [ubuntu/kimi.md](ubuntu/kimi.md).

**(a) ATOM, ours vs amd-cloud:**

| | Same | Different |
|---|---|---|
| Result | **throughput within ±2% and TPOT within ±1% up to 128 users**; `max-num-seqs 2048` fails on both (does not fit in memory) | 256–512 users: 1.05× throughput; `max-num-seqs 1024`: 1.15–1.26× throughput |

**(b) vLLM recipe 1 vs best ATOM, both on our nodes** (recipe vs recipe, not hardware):

| Users | 1 | 4 | 8 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| vLLM recipe 1 / best ATOM, tok/s | 1.56× | 1.39× | 1.24× | 1.05× | 1.01× | 0.97× |

The recipe's speculative decoding helps at low load; at high load both are equal.

**(c) vLLM recipe 2 on our nodes** (`amd-kimi-k3-recipe.pdf`: vLLM v0.29.0, TP8, `max-num-seqs 128`, no speculative decoding). Detail: [ubuntu/kimi-amd-recipe.md](ubuntu/kimi-amd-recipe.md), all Kimi-K3 results on one page: [kimi-summary.md](kimi-summary.md).

- **Our nodes vs AMD's published numbers, AMD's workload (ISL/OSL 128K/1K):** our runs already use vLLM recipe 2 itself (same image, server flags and client settings as the PDF); the published numbers are the PDF's results table. Result: our nodes are the same as AMD's at 8–16 users (1.01×) and faster at 32–128 (1.07–1.10×); at 1–4 users they are slower (0.56–0.82×).
- **Results on our workload (1K/1K, node6101):** 54 tok/s at 1 user (TPOT 18.1 ms), 315 at 8, 1,228 at 64 and 1,735 at 128 (TPOT 71.8 ms), TTFT under 0.5 s up to 128 users. At 256 users throughput stays at 1,747 and TTFT jumps to 73 s, because `max-num-seqs 128` makes half the requests wait.
- The comparison with ATOM and vLLM recipe 1 is in its own file: [ubuntu/kimi-recipe-old-vs-new.md](ubuntu/kimi-recipe-old-vs-new.md).
