# Kimi-K3 on 2 × 8 × B200 — compute and communication analysis

Serving `moonshotai/Kimi-K3` (2.78 T params, 1.56 TB MXFP4 checkpoint) on **2 nodes × 8 B200** via vLLM, **TP=8 × PP=2**. Measured 2026-08-21T17:13:53Z.

> **Why two nodes.** The checkpoint is **1561 GB (1.42 TiB)**. One 8 × B200 node holds **1538 GB (1.40 TiB)** of HBM (8 × 192 GB) — **23 GB short of the weights alone**, before a single byte of KV cache, activation workspace, NCCL buffers or CUDA-graph pool, which together take another ~15–20 GB per GPU. There is no single-node B200 configuration for this model. TP8 shards within each node and PP2 splits the 93 layers across the pair. The MI355X baseline runs the same model on **one** node, because 8 × 288 GB = 2304 GB does fit with room to spare. Every throughput figure below is therefore also reported per GPU and per node.

**Run configuration** (from the server log, not assumed):

| Setting | Value |
|---|---|
| Parallelism | `tensor_parallel_size=8`, `pipeline_parallel_size=2`, DP=1, **EP off** |
| GPUs / nodes | 16 / 2 |
| Quantization | routed MoE experts **MXFP4** (`mxfp4-pack-quantized`, group_size 32); attention, shared experts and dense MLP left at BF16 by the checkpoint's `ignore` list |
| KV cache dtype | fp8 |
| `max_model_len` / `max_num_seqs` | 16384 / 64 |
| `max_num_batched_tokens` | 8192 |
| `gpu_memory_utilization` | 0.9 |
| Prefix caching | **disabled** (disabled is required — KDA recurrent state can't be rebuilt from the paged cache) |
| Workload | ISL/OSL 1024/1024, `--ignore-eos`, concurrency 1→64 |
| Weight load | 10.6 min for 1.42 TiB off shared NFS (~2.5 GB/s effective) |

**Architecture** (parsed from `config.json`): 93 layers — **24 MLA full-attention** + **69 KDA linear-attention**; hidden 7168; MoE with **896 routed experts, top-16 + 2 shared**, routed-expert latent 3584 → 3072.

The parse is **exactly validated**: routed-expert parameters computed from the config (`92 MoE layers × 896 experts × 3 × 3584 × 3072`) come to **2,722,740,830,208**, which is the checkpoint's own MXFP4 (U8) parameter count to the last digit. Total computed ≈ **2.777 T** against the advertised 2.78 T.

> The routed experts live in a **latent space of 3584**, not on the 7168-wide residual stream. That is the difference between 2.72 T and the ~6.4 T a naive `3 × hidden × expert_width` count would predict, and it is why the exact match above is worth stating.

---

## 0. Overview — the short version

**§1 Compute** — **1,696.4 tok/s** at c=64 (20× scaling from c=1, TPOT only 3.2× worse). Achieved **349.5 TFLOP/s aggregate = 21.8/GPU = 1.0% of B200 BF16 peak**. Only 103 B of 2.78 T params activate per token (3.7%).

**§2 Memory** — Per GPU: **97.0 GiB weights** + **59.1 GiB KV pool**. KV decodes exactly: 13,824 B/token = `(kv_lora 512 + rope 64) × 1 B fp8 × 24 MLA layers` — proving only the 24 full-attention layers page KV, and that KV is replicated across TP ranks rather than sharded.

**§3 Bottleneck — HBM traffic, and it is LATENCY-BOUND, NOT BANDWIDTH-BOUND.** Compute 1.0% utilized, NVLink 0.5%, **HBM only ~23%** (1,806.5 GB/s of 8,000) — the bandwidth is there and is going unused. At batch 64 the tokens route independently, so **610 of 896 experts** activate per layer (not 16), yet each expert then sees only **1.7 tokens** — a matrix-*vector* product that cannot keep enough memory requests in flight to saturate HBM. Weight traffic dominates step time, but the ceiling being hit is memory-access latency/occupancy, not bandwidth. The fix follows directly: widen the GEMMs (§3.2).

**§4 Communication** — two paths now, not one. NVLink carries 93 all-reduces/token within each node (4.16 GB/s per GPU, 0.5% of ceiling). **The PP2 boundary additionally puts InfiniBand in the per-token critical path** — 0.026 GB/s across the pair (0.01% of the 8-rail NDR fabric). This is the cost the MI355X run does not pay, and §4.2 sizes it.

**§5** — the levers, and §6 the head-to-head against MI355X.

---

## 1. Computing performance

**Metrics used throughout this report.** Every one is measured by `vllm bench serve`, not derived, unless stated otherwise:

| Term | Stands for | What it measures |
|---|---|---|
| **TTFT** | **Time To First Token** | Latency from sending a request to receiving its *first* output token — i.e. how long the user waits before anything appears. Dominated by **prefill** (processing the whole input prompt). |
| **TPOT** | **Time Per Output Token** | Average latency *between* successive output tokens, after the first. This is the **decode** step time — how fast the answer streams once it has started. Reported as a median over all requests. |
| **Concurrency** | — | Number of independent requests in flight at once. A client-side load setting, not a hardware unit: all 64 requests at c=64 are batched together across the same 16 GPUs in one continuous-batching loop. |
| **Throughput (tok/s)** | — | **Aggregate** output tokens per second across *all* concurrent requests — not what any single user sees (§1.1). |
| **Per-user tok/s** | — | `1000 / TPOT` — just a unit flip: TPOT is ms per token, so `1/TPOT` is tokens per ms and `×1000` makes it tokens per second. The streaming rate one user actually experiences. *Derived*, not measured directly (§1.1). |
| **req/s** | — | Completed requests per second. |

TTFT and TPOT answer different questions and are bound by different things here: TTFT is compute-dense prefill and stays nearly flat with load, while TPOT is memory-bound decode and grows with it (§3.3). Both are quoted as **medians** in the tables below; the raw JSON also carries p99.

| Concurrency | Throughput (tok/s) | tok/s per GPU | TTFT med (ms) | TPOT med (ms) | req/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 86.7 | 5.4 | 225.9 | 11.24 | 0.08 |
| 2 | 165.2 | 10.3 | 217.6 | 11.85 | 0.16 |
| 4 | 280.8 | 17.6 | 233.3 | 13.62 | 0.26 |
| 8 | 475.3 | 29.7 | 234.5 | 15.41 | 0.44 |
| 16 | 801.8 | 50.1 | 234.6 | 18.65 | 0.77 |
| 32 | 1,209.3 | 75.6 | 264.4 | 25.28 | 1.16 |
| 64 | **1,696.4** | 106.0 | 278.1 | 35.89 | 1.64 |

Throughput scales **20×** from c=1 to c=64 while TPOT grows only 3.2×. The sweep stops at 64 because the server was launched with `--max-num-seqs 64`; past it you would measure queueing, not the engine.

### 1.1 Per-user token rate — what one user actually experiences

The table above is **aggregate** throughput, shared across all concurrent requests. A single user does not experience it. What one user sees is the streaming rate of their own answer, **1000 / TPOT** (TPOT = Time Per Output Token, the decode-step latency), and that moves in the *opposite* direction:

*Why `1000 / TPOT`:* TPOT is milliseconds per token, so one over it is tokens per millisecond, and ×1000 gives tokens per second — a user receiving a token every 11.24 ms is receiving 89 tokens per second. It is **per-user** because TPOT is measured within a single request: continuous batching advances many requests in one forward pass, so the engine emits N tokens per step while each user still gets only one. That is the whole reason the two columns diverge — aggregate ≈ N × per-user. Note it is the steady-state rate and **excludes TTFT**: the full wait for an N-token answer is `TTFT + (N-1) × TPOT`.

| Conc | Total tok/s | TPOT med (ms) | **Per-user tok/s** | vs single user |
|---:|---:|---:|---:|---:|
| **1** | 86.7 | 11.24 | **89.0** | _reference_ |
| 2 | 165.2 | 11.85 | 84.4 | -5% |
| 4 | 280.8 | 13.62 | 73.4 | -18% |
| 8 | 475.3 | 15.41 | 64.9 | -27% |
| 16 | 801.8 | 18.65 | 53.6 | -40% |
| 32 | 1,209.3 | 25.28 | 39.6 | -56% |
| **64** | 1,696.4 | 35.89 | **27.9** | -69% |

**Aggregate throughput and single-user speed trade against each other.** Going c=1 → c=64 buys **19.6× aggregate throughput** and costs **3.2× per-user speed** (89.0 → 27.9 tok/s). Both are real; which one matters depends entirely on the workload:

| Optimising for | Run at | Read |
|---|---|---|
| One interactive user, a latency SLO, a long agentic session | **low concurrency** | per-user tok/s |
| Many users at once, cost per token, GPU utilisation | **high concurrency** | total tok/s |

> **This is the caveat on §3.2.** Every lever proposed there raises *aggregate* throughput, and three of the four do it by increasing batch size — which makes single-user speed **worse**. Raising `--max-num-seqs` is the right call for a busy server and the wrong call for one user waiting on one answer.

Note also that **none of the §3.2 levers help at c=1 at all**: they all widen the per-expert GEMM, and with one token in flight there is nothing to widen. The single-user fixes are different ones — speculative decoding / MTP (blocked here, see §5), removing PP (impossible on B200, see §1 preamble), and lower TP. See `notes-concurrency.md`.

### Achieved TFLOP/s

Only **top-16 + 2 of 896** experts fire per token. Active params per token ≈ **103 B** (of 2.78 T — 3.7% activation ratio):

| Component | Active params/token |
|---|---:|
| FFN (top-16 routed + 2 shared + latent projections, 92 layers) | 66.8 B |
| Attention (24 MLA + 69 KDA) | 33.9 B |
| Embedding / lm_head | 2.3 B |

At 2 FLOP per active param per token:

| Concurrency | Aggregate TFLOP/s | Per GPU | % of B200 BF16 peak (2,250) |
|---:|---:|---:|---:|
| 1 | 17.9 | 1.12 | 0.05% |
| 64 | 349.5 | 21.85 | 0.97% |

**Decode is nowhere near compute-bound** — barely 1% of peak. Autoregressive decode issues one token per sequence per step, so every weight matrix is used for a single narrow GEMV-like operation. This is a *memory-bandwidth* regime, quantified in §3.

> Caveat: the active-parameter figure is config-derived. The routed-expert term is exact (it reproduces the checkpoint's MXFP4 count to the digit); the KDA term is an approximation from the config dimensions and dominates the attention row. Treat the 103 B figure as ±10%. The conclusion (decode is ~1% of peak) has far too large a margin to be affected.

---

## 2. GPU memory usage

Measured per rank at load time, straight from the server log:

```
total_gpu=?GiB  utilization=0.9  budget=?GiB
weights=97.03GiB  non_torch=?GiB  act_peak=?GiB  kv=59.14GiB
kv_tokens=6435401  kv_blocks=?
```

Per GPU (× 16 for the 2-node pair):

| Component | Per GPU | Job total | What it is |
|---|---:|---:|---|
| Model weights + framework | 97.0 GiB | 1,552.5 GiB | The TP8×PP2 shard — dominated by MXFP4 experts |
| **KV cache pool** | 59.1 GiB | 946.2 GiB | Allocated up front from what's left |

**It does not load only weights.** Weights are ~97.0 GiB/GPU (1454 GiB of checkpoint sharded 16 ways ≈ 91 GiB, consistent), and a **59.1 GiB KV pool per GPU** is carved out on top.

### KV cache details

Measured `kv_bytes / kv_tokens` = **9,867 B per token per GPU**.

Predicted from the architecture: MLA stores a *compressed latent* — `kv_lora_rank(512) + qk_rope_head_dim(64)` = 576 values/token/layer, at fp8 = 576 B, × **24 full-attention layers** = **13,824 B**.

Measured is 0.71× predicted — the gap is worth chasing before trusting the conclusions below. Candidates: KDA state counted into the same pool, or a block-padding effect.

1. **Only the 24 MLA layers consume paged KV.** The 69 KDA layers keep a fixed-size *recurrent state* per request instead of a growing cache — that is the point of linear attention, and it is why a 93-layer model has the KV footprint of a 24-layer one.
2. **The KV cache is replicated, not sharded, across TP ranks.** MLA's latent is shared across heads, so sharding it would mean re-gathering it every step. Replication trades otherwise-idle HBM for zero communication.

Capacity: **6,435,401 tokens** per GPU. The benchmark used 64 seqs × ~2048 tokens = **131,072 tokens, 2.0% of the pool**. KV was never remotely a constraint.

---

## 3. What is the bottleneck?

**Intra-GPU HBM traffic.** Not compute, not either interconnect.

**But be precise about which HBM limit: this is LATENCY-BOUND, NOT BANDWIDTH-BOUND.** HBM sits at only ~23% of peak, so the box has bandwidth to spare. What binds is that the MoE decode GEMMs are too narrow to keep enough concurrent memory requests in flight to use it (§3.2). Saying "HBM-bandwidth-bound" would point at the wrong fix — buying faster memory would not help; widening the GEMMs would.

| Resource | Demand at c=64 | B200 capability | Utilization |
|---|---:|---:|---:|
| Compute | 21.8 TFLOP/s per GPU | 2,250 TFLOP/s BF16 | **0.97%** |
| **HBM bandwidth** | **~1,807 GB/s per GPU** | 8,000 GB/s | **~23%** |
| NVLink (GPU↔GPU, intra-node) | 4.16 GB/s per GPU | ~900 GB/s (per direction) | **~0.46%** |
| InfiniBand (PP stage boundary) | 0.026 GB/s | 400 GB/s (8 rails NDR) | **~0.01%** |

### 3.1 Why HBM — the MoE batching mechanism

At **batch 1**, only 16+2 experts fire per layer. At **batch 64** the tokens route independently, so the expected number of *distinct* experts touched per layer is `E × (1 − (1−1/E)^(B·topk))` = **610 of 896**.

| Batch | Experts fired/layer | Tokens per expert | Weight bytes/step (job) | per GPU |
|---:|---:|---:|---:|---:|
| 1 | 16 | 1.0 | 26 GB | 2 GB |
| 8 | 119 | 1.1 | 193 GB | 12 GB |
| 64 | 610 | 1.7 | 985 GB | 62 GB |
| 128 | 805 | 2.5 | 1300 GB | 81 GB |
| 256 | 887 | 4.6 | 1432 GB | 89 GB |
| 512 | 896 | 9.1 | 1446 GB | 90 GB |
| 1024 | 896 | 18.3 | 1446 GB | 90 GB |

This is the defining property of sparse MoE: **compute grows with batch, but weight traffic grows much faster** until nearly every expert is touched every step. From batch ~512 upward the weight bytes **plateau** — every additional token is then nearly free in bandwidth terms.

MXFP4 is what makes this tractable. At BF16 the same reads would be **4× larger**, exceeding HBM bandwidth outright and making the model bandwidth-*starved* rather than merely bandwidth-dominated.

### 3.2 How to improve it

HBM is the binding resource at ~23%, but the box is not delivering all the HBM it has. The shortfall has a specific cause: with 610 experts fired across 64 tokens, each expert sees only **1.7 tokens** — a matrix-*vector* product, which cannot issue enough concurrent memory requests to saturate HBM.

**This is the crux: the workload is latency-bound, not bandwidth-bound.** Its *volume* of weight traffic is what dominates step time, but it is nowhere near the bandwidth ceiling — a GEMV has too little memory-level parallelism to fill the pipe. More bandwidth would buy nothing; more tokens per expert buys everything.

Ranked levers:

1. **Raise `--max-num-seqs`** (64 → 256+). Biggest lever, costs nothing, and the KV memory is already provisioned (§2). This is the same conclusion the MI355X run reached, and for the same reason.
2. **Speculative decoding / MTP** — verifies several tokens per weight read, widening the GEMM exactly as a larger batch does. Note the vLLM recipe **gates DSpark off the `multi_node_tp_pp` profile** (it does not compose with pipeline parallelism yet, vllm-project/vllm#50098), so on this 2-node layout it is unavailable — a real cost of needing PP that the single-node MI355X layout does not pay.
3. **Expert parallelism** — fewer, whole expert reads per GPU, paid for with all-to-all over the near-idle interconnect. On MI355X this was tested and is **unsupported** (ATOM raises `NotImplementedError` for EP with the MXFP4 SiTUv2 kernel). On B200 it is worth testing separately; it is deliberately **off** here because the MI355X baseline has it off, and arm A exists to hold everything but the hardware constant.
4. **Prefill/decode disaggregation** — the recipe ships a `pd_cluster` profile.

### 3.3 Why TTFT is flat but TPOT rises

TTFT moves 226 → 278 ms (1.2×) across a 64× concurrency increase, because prefill is genuinely compute-dense (1024 tokens/request in parallel) and has headroom. TPOT rises 3.2× because decode adds weight-read traffic per step as more experts activate. Different regimes — further confirmation that decode is bandwidth-limited.

---

## 4. Data communication analysis

Three paths carry traffic here, where the single-node MI355X run had one.

### 4.1 Intra-node GPU↔GPU (NVLink) — activations only

With **TP=8 and EP disabled**, every expert is sharded across the 8 GPUs of its node, so there is **no expert-routing all-to-all**. The only cross-GPU traffic inside a node is TP activation reduction:

| Property | Value |
|---|---|
| Collective | **all-reduce** (NCCL), 2 per layer |
| Layers per pipeline stage | 93/2 = 46.5 |
| Count per token per node | 2 × 46.5 = **93** |
| Payload per call per token | `hidden_size × 2 B` = **14.0 KB** |

| Concurrency | Steps/s | Payload/step | Wire bytes/step¹ | Sustained per GPU |
|---:|---:|---:|---:|---:|
| 1 | 89.0 | 1.3 MB | 2.3 MB | **0.21 GB/s** |
| 64 | 27.9 | 85.3 MB | 149.3 MB | **4.16 GB/s** |

¹ busbw convention: an all-reduce moves `2(N−1)/N × payload` on the wire.

At **4.16 GB/s against a ~900 GB/s per-direction ceiling (0.46%)**, NVLink is almost entirely idle.

**What is NOT transferred:** weights (resident per GPU), KV cache (replicated), gradients (inference), expert tokens (EP off).

#### The message-size regime — why "1% utilized" understates the cost

| | Per all-reduce call |
|---|---|
| Payload at c=1 | **14.0 KB** |
| Payload at c=64 | **896 KB** |

A step-time budget at c=64 (step = 35.9 ms, 93 all-reduces):

| Assumption | All-reduce time/step | Share of step |
|---|---:|---:|
| Pure bandwidth, zero overhead | 0.17 ms | 0.5% |
| + 5 µs fixed latency per call | 0.63 ms | 1.8% |
| + 10 µs fixed latency per call | 1.10 ms | 3.1% |
| + 20 µs fixed latency per call | 2.03 ms | 5.6% |

These calls are **latency-dominated, not bandwidth-dominated**: a few microseconds each compounds into milliseconds across 93 serialized collectives per step. The utilization percentage is a floor on the cost, not an estimate of it.

> Not directly measured: there is no per-call NCCL timing from this run, so the latency rows are illustrative arithmetic over a plausible range. Confirming the real figure needs `--profile` or `NCCL_DEBUG=INFO` timing.

### 4.2 Inter-node (InfiniBand) — the pipeline boundary

**This path has no counterpart in the MI355X run**, which is single-node. It exists here only because the model does not fit in one node.

With PP=2, the 93 layers are split into 2 stages. Every step, stage 0 sends its hidden states to stage 1 over IB and (for the next microbatch) receives. The payload is small — hidden state, not weights:

| Property | Value |
|---|---|
| Stage boundaries | 1 |
| Bytes per token per boundary | `hidden_size × 2 B` = 14.0 KB |
| Bytes per step at c=64 | 0.92 MB |
| Sustained | **0.026 GB/s** |
| Fabric capability (8 rails NDR, per node per direction) | 400 GB/s |
| Utilization | **0.006%** |

Measured fabric health on these nodes, for context (`../b200-nodes/notes.md`, 2026-08-12): `ib_write_bw` GPU→GPU at **395.5 Gb/s** — NDR line rate — and NCCL `sendrecv` at **48.4 GB/s per pair** at 8 GPUs/node. The fabric is not the problem.

**But bandwidth is again the wrong lens, and here it matters more.** The PP cost is not the bytes; it is:

1. **Latency in the critical path.** Every token must traverse stage 0, cross the network, then traverse stage 1. One inter-node hop is added to every decode step — an RDMA write plus synchronization, on the order of single-digit microseconds, against a 35.9 ms step. Small, but it is pure serial addition.
2. **The pipeline bubble.** vLLM splits the running batch into microbatches to keep both stages busy; whatever it cannot overlap is idle GPU time on one stage or the other. This is the real PP tax and it is **not visible in the byte counts above**.
3. **Lost features.** DSpark speculative decoding is gated off this profile entirely.

A clean measurement of the bubble needs a stage-resident profile (per-stage step timings), which this run does not collect. What can be said from these data: the PP boundary is **not bandwidth-limited**, so if PP costs throughput here it costs it through bubbles and latency, not through the wire.

### 4.3 Intra-GPU (HBM) — dominated by weights

Per decode step at c=64, per GPU:

| Traffic | Bytes/step | Share |
|---|---:|---:|
| **Expert weights (MXFP4)** | 61.59 GB | 92.2% |
| Attention + shared + dense weights | 3.25 GB | 4.9% |
| KV cache read | 1.81 GB | 2.7% |
| Activations (read+write) | 0.17 GB | 0.3% |

The asymmetry is stark: **HBM moves ~67 GB/step while NVLink moves ~0.15 GB and IB ~0.0009 GB.** Optimization effort belongs on the memory side.

---

## 5. Further discussion

**1. Two nodes is a property of the model, not a choice.** 1561 GB of weights against 1538 GB of node HBM — 23 GB short. The consequence is not just "more GPUs": it forfeits DSpark speculative decoding (gated off `multi_node_tp_pp`), adds a pipeline bubble, and halves the per-GPU weight residency. A B300 node (8 × 268 GB = 2144 GB) would hold it single-node; a B200 node cannot.

**2. `max_num_seqs=64` is the binding limit, not hardware.** KV was ~2.0% used and compute ~1%.

**3. Prefix caching is disabled for correctness, and it costs real throughput.** KDA's recurrent state is per-request and cannot be reconstructed from the paged MLA cache. In workloads with shared prefixes this forfeits a large win that non-KDA models get for free — an architectural trade, not a tuning oversight. (The vLLM Blackwell baseline turns prefix caching *on*; it is turned off here both for correctness and to match the MI355X run.)

**4. The hybrid attention design is what makes 2.78 T fit at all.** Only 24 of 93 layers keep a growing KV cache.

**5. Load time was 10.6 minutes** for 1.42 TiB off shared NFS (~2.5 GB/s effective), with `--load-format fastsafetensors`.

---

## 6. B200 vs MI355X — head to head

> **Read this section carefully — it is not an equal-hardware comparison.** MI355X serves this model on **one node with 8 GPUs**; B200 needs **two nodes and 16 GPUs**, because the checkpoint does not fit in 8 × 183 GB. Total throughput therefore compares two different amounts of hardware. The per-GPU and per-node columns are the ones that carry meaning, and even those are confounded by a different engine (vLLM vs ATOM) and a different parallelism (TP8×PP2 vs TP8). Treat this as a **system-level capability comparison**, not a chip-vs-chip benchmark.

### 6.1 What was held constant, and what could not be

| | MI355X | B200 | Matched? |
|---|---|---|---|
| Model | Kimi-K3 MXFP4 | Kimi-K3 MXFP4 | ✅ |
| ISL / OSL | 1024 / 1024 | 1024 / 1024 | ✅ |
| `max_model_len` | 16384 | 16384 | ✅ |
| `max_num_seqs` | 64 | 64 | ✅ |
| KV dtype | fp8 | fp8 | ✅ |
| Prefix caching | off | off | ✅ |
| Expert parallelism | off | off | ✅ |
| Concurrency sweep | 1→64 | 1→64 | ✅ |
| **Nodes / GPUs** | **1 / 8** | **2 / 16** | ❌ *forced* |
| **Parallelism** | **TP8** | **TP8×PP2** | ❌ *forced* |
| **Engine** | **ATOM** | **vLLM** | ❌ *ATOM is ROCm-only* |

### 6.2 Hardware

| | MI355X | B200 | Ratio (B200/MI355X) |
|---|---:|---:|---:|
| HBM per GPU | 288 GB | 192 GB | 0.67× |
| HBM per node (8 GPUs) | 2304 GB | 1538 GB | 0.67× |
| HBM bandwidth | 8,000 GB/s | 8,000 GB/s | 1.00× |
| BF16 dense peak | 2,500 TFLOP/s | 2,250 TFLOP/s | 0.90× |
| GPU↔GPU link | Infinity Fabric (xGMI) | NVLink 5 | — |
| link, per direction | 537 GB/s | 900 GB/s | 1.68× |
| **Holds Kimi-K3 on one node?** | **yes** — 2304 GB vs 1561 GB of weights, 743 GB spare | **no** — 1538 GB vs 1561 GB, **23 GB short** | — |

**HBM capacity, not bandwidth or FLOPs, is the axis that decides this workload's layout.** The two parts have identical HBM bandwidth and B200 has 90% of MI355X's BF16 peak — neither of those decides anything here. Capacity does: 288 GB/GPU vs 192 GB/GPU is 1.50×, and that single ratio is the difference between one node and two.

### 6.3 Throughput, point by point

| Conc | MI355X tok/s | B200 tok/s | MI355X/B200 | B200/MI355X | MI355X tok/s/GPU | B200 tok/s/GPU | per-GPU ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 46.1 | 86.7 | 0.53× | 1.88× | 5.8 | 5.4 | 0.94× |
| 2 | 87.0 | 165.2 | 0.53× | 1.90× | 10.9 | 10.3 | 0.95× |
| 4 | 154.2 | 280.8 | 0.55× | 1.82× | 19.3 | 17.6 | 0.91× |
| 8 | 288.0 | 475.3 | 0.61× | 1.65× | 36.0 | 29.7 | 0.83× |
| 16 | 500.9 | 801.8 | 0.62× | 1.60× | 62.6 | 50.1 | 0.80× |
| 32 | 824.0 | 1,209.3 | 0.68× | 1.47× | 103.0 | 75.6 | 0.73× |
| 64 | 1,258.5 | 1,696.4 | 0.74× | 1.35× | 157.3 | 106.0 | 0.67× |

| Headline | MI355X | B200 | MI355X/B200 |
|---|---:|---:|---:|
| tok/s at c=1 *(single request)* | 46.1 | 86.7 | 0.53× |
| tok/s at c=1 **per GPU** | 5.8 | 5.4 | 1.06× |
| TPOT at c=1 (ms, lower better) | 21.48 | 11.24 | 1.91× |
| Peak tok/s (c=64) | 1,258.5 | 1,696.4 | 0.74× |
| Peak tok/s **per GPU** | 157.3 | 106.0 | 1.48× |
| Peak tok/s **per node** | 1,258.5 | 848.2 | 1.48× |
| GPUs to serve the model | 8 | 16 | 0.50× |

### 6.4 Latency

| Conc | MI355X TTFT | B200 TTFT | MI355X TPOT | B200 TPOT | MI355X per-user tok/s | B200 per-user tok/s | **B200/MI355X** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 224.9 | 225.9 | 21.48 | 11.24 | 46.6 | 89.0 | **1.91×** |
| 2 | 251.5 | 217.6 | 22.58 | 11.85 | 44.3 | 84.4 | **1.91×** |
| 4 | 256.5 | 233.3 | 24.98 | 13.62 | 40.0 | 73.4 | **1.83×** |
| 8 | 257.5 | 234.5 | 27.02 | 15.41 | 37.0 | 64.9 | **1.75×** |
| 16 | 261.7 | 234.6 | 31.21 | 18.65 | 32.0 | 53.6 | **1.67×** |
| 32 | 273.9 | 264.4 | 37.83 | 25.28 | 26.4 | 39.6 | **1.50×** |
| 64 | 285.6 | 278.1 | 49.91 | 35.89 | 20.0 | 27.9 | **1.39×** |

TPOT is the metric where the PP2 boundary would show up if it costs anything: it is per-decode-step latency, and the inter-node hop plus any pipeline bubble lands there directly. TTFT is prefill-dominated and less sensitive to it.

**B200 is faster per user at every concurrency — and 1.91× faster for a single user** (89.0 vs 46.6 tok/s at c=1). This is the one comparison B200 wins outright, and §6.3's aggregate view hides it completely.

The two results are not in conflict; they answer different questions:

- *How many tokens per second per unit of silicon?* → **MI355X**, 1.48× per GPU (§6.3).
- *How fast does one user's answer stream?* → **B200**, 1.91× (here).

B200's lower per-token latency wins the second question and survives even the PP2 pipeline penalty. See `notes-concurrency.md` for the full treatment, including why none of the §3.2 levers improve the single-user case.

### 6.5 Where each system's headroom is

| Resource (at peak concurrency) | MI355X | B200 |
|---|---:|---:|
| Compute utilization | 1.30% | 0.97% |
| HBM utilization | ~32% | ~23% |
| GPU↔GPU link utilization | ~1.11% | ~0.46% |
| Inter-node utilization | n/a (single node) | ~0.006% |
| Experts fired per layer | 610 / 896 | 610 / 896 |
| Tokens per expert | 1.7 | 1.7 |

**Both systems land in the same regime and hit the same wall**: ~1% of compute, ~1% of interconnect, and a memory system doing most but not all of what it can, because each expert's weight matrix is read to serve barely more than one token. The lever on both is `max_num_seqs`. That the two agree on the diagnosis, from different vendors' silicon and different serving engines, is the strongest evidence that the finding is about the *model*, not about either box.

### 6.6 Memory footprint

| | MI355X (TP8, 1 node) | B200 (TP8×PP2, 2 nodes) |
|---|---:|---:|
| Weights per GPU | 190.4 GiB | 97.0 GiB |
| KV pool per GPU | 57.7 GiB | 59.1 GiB |
| KV capacity per GPU | 4,221,440 tok | 6,435,401 tok |
| KV bytes/token | 13,824 B | 13,824 B |
| Total HBM committed | 1,984.5 GiB | 2,498.7 GiB |

Per-GPU weight residency is roughly halved on B200 — the same checkpoint spread over twice as many GPUs. That is why B200's per-GPU weight *traffic* is also roughly halved in §3, and why per-GPU throughput comparisons must be read alongside the GPU count, not instead of it.

### 6.7 What this comparison does and does not establish

**Does:**

- MI355X serves Kimi-K3 on **one** node; B200 needs **two**. For a 2.78 T MXFP4 frontier model, HBM capacity per node is the deciding specification.
- Both reach the same bottleneck (HBM, via thin MoE GEMMs) and the same lever (`max_num_seqs`), independently.
- The inter-node fabric is **not** the limiting factor on the B200 pair: the PP boundary uses a fraction of a percent of an 8-rail NDR fabric that measures at line rate.

**Does not:**

- Establish a per-chip performance ratio. Engine (vLLM vs ATOM), parallelism (PP2 vs none) and GPU count all differ; no single ratio isolates the silicon.
- Say anything about B200 with a model that *does* fit in one node — that is a different and more favourable configuration for B200, and it is not this measurement.
- Measure the pipeline bubble, which needs per-stage profiling this run does not collect.

---

## 7. Cross-check against SemiAnalysis InferenceX

SemiAnalysis publish continuous Kimi-K3 benchmarks at <https://inferencex.semianalysis.com>, and open-source the harness (Apache-2.0, `github.com/SemiAnalysisAI/InferenceX`). Their configs are in `../semianalysis-ref/`. Their **B200 recipe uses TP8 × PP2 on 2 nodes — the same layout as this run**, which makes the methodology directly checkable.

### 7.1 What they independently confirm

| Fact | **Applies to** | Ours (measured) | Theirs (stated in config) |
|---|---|---|---|
| Checkpoint size | **both** | 1,560,936,091,448 B = 1.561 TB / 1.420 TiB, 96 shards | *"1.561 TB decimal (1.420 TiB, 96 safetensors)"* |
| Does not fit one 8×B200 node | **B200** | 1561 GB vs 1538 GB — 23 GB short | *"does not fit one 8xB200 node, so TP8 shards … and PP2 splits the 93 layers"* |
| Layout forced to TP8 × PP2, 16 GPUs | **B200** | yes | `tensor-parallel-size: 8`, `pipeline-parallel-size: 2`, `agg_nodes: 2` |
| `gpu-memory-utilization` 0.90 not 0.95 | **B200** | 0.90 | 0.90, and the *same reason*: *"the flashinfer trtllm MXFP4 MoE kernel allocates a ~1.6 GiB runtime workspace OUTSIDE vLLM's memory pool … at 0.95 a 178 GiB B200 … OOMs"* |
| Usable HBM per GPU | **B200** | 178.35 GiB (nvidia-smi) | *"a 178 GiB B200"* |
| Fits on ONE node at TP8 | **MI355X** | yes, 288 GB/GPU | *"~195 GB/GPU across 8 GPUs of the 288 GB part; TP=4 … cannot load"* |
| Expert parallelism off | **both** | EP off | *"Plain TP (NOT TEP): expert parallelism is deliberately off"* |
| Image, load format, autotune, batched-token cap | **B200** | `vllm/vllm-openai:kimi-k3`, `fastsafetensors`, autotune off, 8192 | identical on all four |

Two independent teams, different clusters, same conclusions — including the exact checkpoint byte count and the non-obvious 0.90 memory-utilisation workaround.

### 7.2 What differs — why the numbers are NOT directly comparable

| Dimension | **Applies to** | Ours | SemiAnalysis | Effect |
|---|---|---|---|---|
| **Workload** | **both** | fixed ISL/OSL 1024/1024, `--ignore-eos`, random synthetic | **AgentX agentic trace replay**, real multi-turn traces, 1M+ context | **largest difference.** Their traces have long shared prefixes and huge context; ours is a fixed, cache-hostile synthetic shape |
| **Prefix caching** | **both** | **off** | **on** (default, kept for trajectory reuse) | theirs reuses KV across turns; ours never does. Big throughput swing on agentic traffic |
| `max-model-len` | **both** | 16384 | native **1M** (unset) | theirs pays a far larger KV footprint per sequence |
| `max-num-seqs` | **B200** | **64** (fixed) | let vLLM choose | ours deliberately caps the batch; §3.2 shows that cap is what binds our throughput |
| Benchmark client | **both** | `vllm bench serve` | `aiperf` + trace replay | different measurement harness |
| **Spec decoding (MTP)** | **MI355X** | **off** | **DSpark MTP on** (`SPEC_NUM_TOKENS 2`) | their MI355X arm gets a lever ours does not use — see §7.4 |
| `max-num-seqs` | **MI355X** | 64 | 128 | their MI355X runs a deeper batch |
| Serving engine | **MI355X** | ATOM (our baseline) | vLLM ROCm *and* an ATOM variant | we compare ATOM-vs-vLLM; they run both |

**Precision is the same, despite the labels.** Their config says `precision: "fp4"` and ours says MXFP4 — the same thing. Both serve the native `moonshotai/Kimi-K3` MXFP4 checkpoint (`mxfp4-pack-quantized`, 4-bit routed experts with e8m0 scales); neither re-quantises. "FP4" on their dashboard is the checkpoint's own format, not a separate NVFP4 conversion.

### 7.3 Per-user tok/s — theirs vs ours

Retrieved live from their public API (`/api/v1/benchmarks?model=Kimi-K3`); raw JSON and an extracted CSV are in `../semianalysis-ref/`. Their **"Interactivity"** metric is exactly our per-user tok/s — verified against their own fields: `mean_tpot` 0.00454 s → 1/0.00454 = 220.3 = their `mean_intvty`. Same definition, `1 / TPOT`.

| Conc | **Ours** B200 TP8×PP2 (no spec) | **Theirs** B200 TP8×PP2 (no spec) | **Theirs** B200 +MTP | **Theirs** MI355X vLLM +MTP | **Theirs** MI355X ATOM +MTP | **Ours** MI355X ATOM (no spec) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **89.0** | 81.9 | **221.7** | 84.0 | 127.2 | 46.6 |
| 2 | 84.4 | 74.8 | 219.3 | — | — | 44.3 |
| 4 | 73.4 | 54.1 | 203.3 | 62.5 | 88.7 | 40.0 |
| 8 | 64.9 | 21.1 | 154.1 | 41.4 | 64.7 | 37.0 |
| 10 | — | — | — | 35.7 | 62.7 | — |
| 16 | 53.6 | 8.6 | 77.7 | — | — | 32.0 |
| 32 | 39.6 | 3.8 | 46.5 | — | — | 26.4 |
| 64 | 27.9 | — | — | — | — | 20.0 |

**"no spec" = plain decoding, one token per forward pass. "+MTP" = speculative decoding on: a draft model proposes several tokens and the big model verifies them in one pass.**

> ⚠️ **Every one of their MI355X runs uses MTP — they publish no MI355X result without it — so their MI355X column can never be compared against their B200 no-spec column on equal footing.** (Verified in their API data: all 8 MI355X records are `spec_method: mtp`; B200 has both, 9 `none` and 6 `mtp`.)

Units: output tokens/s delivered to a single request. Theirs are `median_intvty`; ours are `1000 / median TPOT`. **Read across rows with care — see §7.2.** The columns differ in workload (agentic traces vs fixed 1024/1024), prefix caching (on vs off) and context (1M vs 16K), so this is not a like-for-like ranking.

#### What "MTP" is — and it is software, not silicon

**Short answer: yes — MTP is an inference *software* technique, not a hardware capability.** It says nothing about how fast the GPU is. Any vendor's hardware can run it once the software stack supports it. So the MTP columns above measure **software maturity, not silicon**, and must not be read as a hardware comparison.

**MTP = Multi-Token Prediction**, a form of **speculative decoding**:

- **Without it:** one forward pass = **one** token. Every activated expert's weights are read from HBM to emit that single token (~2 GB per GPU at c=1). The weight read is the cost, not the arithmetic.
- **With it:** a small draft model proposes the next **N** tokens; the full model **verifies all N in one pass** — one weight read instead of N. Correct guesses are kept, the first wrong one and everything after is redone.
- **Output is bit-identical** to normal decoding. Pure speedup, no quality tradeoff.

**Why it helps so much here:** decode is *latency-bound, not bandwidth-bound* (§3). Verifying N tokens at once makes each expert GEMM **N wide instead of 1** — the same widening `max-num-seqs` buys, but **without needing other users**. That is why it is the only §3.2 lever that helps a single user (§1.1).

Measured gain in their B200 data (MTP vs no-spec, same hardware):

| | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| B200 per-user tok/s, **no spec** | 81.9 | 21.1 | 3.8 |
| B200 per-user tok/s, **MTP** | 221.7 | 154.1 | 46.5 |
| ratio | **2.7×** | **7.3×** | **12.2×** |

> **Note the trend is the opposite of what simple theory predicts.** Speculative decoding should matter *most* at low concurrency and fade as batching widens the GEMMs on its own — yet here the ratio grows from 2.7× to 12.2×. The likely cause is that the ratio is not a clean MTP-only A/B: the no-spec column collapses steeply (81.9 → 3.8) under long-context agentic traces, and their MTP rows come from a different recipe family than the `agg-b200-tp8pp2-agentic.yaml` config cross-checked in §7.1. Treat the **c=1 figure (2.7×) as the trustworthy one** and the high-concurrency ratios as confounded.

Ceiling is draft length × acceptance rate — never the full N, since not every guess is accepted.

**Two facts that matter for reading the table:**

1. **Kimi-K3 has no built-in MTP head** (`num_nextn_predict_layers = 0`). The gain comes from **DSpark**, a separate speculator model. "MTP" names the technique, not a model feature.
2. **B200 cannot use it in this configuration** — DSpark does not compose with pipeline parallelism, and PP is mandatory on B200 because the model does not fit one node. MI355X fits on one node, needs no PP, and gets MTP for free.

> So the MTP gap is *caused* by a hardware limit (memory capacity forcing PP) but is not itself *a measure of* hardware speed. A B200 with enough memory per node — or a vLLM release that composes DSpark with PP — would get the same ~2.7×.

**What survives those caveats:**

1. **Our B200 no-spec numbers are in the same band as theirs at low concurrency** (89.0 vs 81.9 at c=1) — an independent sanity check that our TP8×PP2 setup is performing normally, not misconfigured.
2. **Their curve falls off far faster than ours** (81.9 → 3.8 by c=32, vs our 89.0 → 39.6). Expected: their agentic traces carry vastly longer contexts, so per-step work grows with concurrency in a way our fixed 1024/1024 shape does not.
3. **MTP is worth ~2.7× at c=1 on B200** (221.7 vs 81.9) in their own data, same hardware and layout. That is the single largest lever in this entire table — and §7.4 explains why it is not available on the TP8×PP2 layout the model forces on B200. Their MTP B200 rows come from a different recipe family than the `agg-b200-tp8pp2-agentic.yaml` we cross-checked.
4. **On MI355X, engine choice is worth ~1.5×** (ATOM 127.2 vs vLLM 84.0 at c=1, both with MTP). Our MI355X baseline is ATOM *without* spec decoding at 46.6, so the gap to their 127.2 is mostly MTP plus a newer ATOM build.

> The honest headline: **on equal footing (no spec decoding, low concurrency) B200 and MI355X land far closer than either vendor's best-configured number suggests, and the biggest single differentiator in the whole table is MTP — a software feature, not silicon.**

### 7.4 How to enable MTP — B200 and MI355X

**Correction to an earlier claim in this report:** spec decoding does *not* require single-node TP. That was inferred from the upstream vLLM recipe gating DSpark off `multi_node_tp_pp` — a **recipe default, not an engine limit**. SemiAnalysis's own B200 recipe runs DSpark with `pipeline-parallel-size: 2`. **MTP is already built into our image** (`vllm/vllm-openai:kimi-k3`) — the `dspark` method, `KimiK3MTP`, and `TOKENSPEED_MLA` are all present. No new package. Both platforms use the **same speculator model**, `Inferact/Kimi-K3-DSpark`.

| Step | MI355X (1 node, TP8) | B200 (2 nodes, TP8×PP2) |
|---|---|---|
| 1. Download the speculator | `Inferact/Kimi-K3-DSpark` | *same* |
| 2. Apply the `pard_token` shim | not needed (single-node TP loads it directly) | **required** — the checkpoint's `mask_token_id` must be aliased to `pard_token` in a local copy of `config.json`, or the config fails to load |
| 3. Extra server flags | none beyond `--speculative-config` | `--decode-context-parallel-size 8 --dcp-comm-backend a2a --attention-backend TOKENSPEED_MLA` |
| 4. `--speculative-config` | `{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":2,"method":"dspark","attention_backend":"TRITON_MLA"}` (SemiAnalysis's `_mtp` recipe) | same JSON, `attention_backend` → `TOKENSPEED_MLA`, `num_speculative_tokens` 7 in their B200 recipe |

**MI355X is simpler because it needs no PP.** One node holds the whole model, so the speculator drops straight into the existing TP8 launch. **B200 needs the shim and the extra DCP flags because of PP2** — those exist to keep the speculator's draft/verify state consistent across the pipeline stage boundary.

**Neither was attempted here** — the speculator is not downloaded (`HF_HUB_OFFLINE=1`). Given the ~2.7× per-user gain at c=1 measured in §7.3, this is the single highest-value follow-up available — bigger than any lever in §3.2.

---

## Source data

| What | Where |
|---|---|
| B200 sweep (per-concurrency JSON) | `logs/kimi_base_20260821_130024/sweep` |
| B200 server log | `logs/kimi_base_20260821_130024/server/vllm_server.log` |
| MI355X sweep | `/orcd/data/orcd/022/benchmarks/amd-benchmarks/amd-cloud/logs/atom/sweep_20260814_164903` |
| MI355X server log | `/orcd/data/orcd/022/benchmarks/amd-benchmarks/amd-cloud/logs/atom/server_20260814_164506/atom_server.log` |
| Model config | `/orcd/compute/orcd/025/models/Kimi-K3/config.json` |
| MI355X published report | `/orcd/data/orcd/022/benchmarks/amd-benchmarks/amd-cloud/results/kimi-k3-base.md` |

Derived figures (active params, FLOP/s, HBM / NVLink / IB volumes) are computed from measured throughput plus the parsed architecture, **using the same formulas for both systems** so the columns are comparable. Memory tables are read from each server's own log. Where this report's derived numbers differ from the published MI355X report, it is because that report's active-parameter estimate omitted the MoE latent projections; this one is validated against the checkpoint's exact MXFP4 parameter count.

---

## Terminology — HBM, NVLink, InfiniBand

Three data paths, and the bottleneck analysis turns on telling them apart.

**HBM** — the GPU's own on-package memory, where weights, KV cache and activations live. **Intra-GPU**: one GPU, no other GPU involved. Every weight read is a read from HBM. B200: 192 GB per GPU at 8,000 GB/s.

**NVLink** — the GPU↔GPU interconnect inside one node, NVIDIA's counterpart to AMD's xGMI. **Intra-node**. B200: 1,800 GB/s bidirectional per GPU (~900 GB/s per direction, the figure a ring all-reduce sees). Carries activation all-reduces only (§4.1).

**InfiniBand** — the **inter-node** fabric. Present in this run and absent from the MI355X one. 8 rails × 400 Gb/s NDR per node, measured at 395.5 Gb/s GPU→GPU. Carries the pipeline stage boundary (§4.2).

**Why the distinction decides everything.** HBM is roughly an order of magnitude faster per GPU than NVLink, so the instinct is that HBM can never be the constraint. That is backwards here: HBM moves ~67 GB per step while NVLink moves ~0.15 GB and IB ~0.0009 GB. The *slower* links are the idle ones.
