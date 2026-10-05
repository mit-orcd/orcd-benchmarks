# ATOM serving benchmark — MI355X

Engine: [ATOM](https://github.com/ROCm/ATOM) (AITER-optimized, vLLM-like) on 8 x MI355X (gfx950), ROCm 7.2.4. Workload: ISL/OSL 1024/1024, `--ignore-eos`, saturating request rate.

Unlike Parts A-C this measures **inference serving**, not raw FLOPS or fabric bandwidth. There is no `dell-cloud/` baseline for this suite — compare against ATOM's public dashboard, not against this repo.

## Summary — three models at a glance

| Tier | Model | Params (total / active) | On disk | TP | Peak tok/s | @ conc | TTFT med @c=1 (ms) | TPOT med @c=1 (ms) | Knee |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tier 1 | `Qwen3-8B-FP8` | 8 B / 8 B | 8.9 GB | 1 | **14,988.0** | 256 | 25.3 | 5.44 | none in range |
| tier 2 | `Llama-3.1-70B-Instruct-FP8` | 70 B / 70 B | 68 GB | 8 | **9,757.0** | 256 | 104.3 | 7.30 | none in range |

*Active* params are what actually fire per token — identical to total for a dense model, but only ~3% of total for Kimi-K3's MoE. That distinction drives most of the throughput differences below.

## The three models

### Tier 1 — `Qwen3-8B-FP8`

[`Qwen/Qwen3-8B-FP8`](https://huggingface.co/Qwen/Qwen3-8B-FP8) · `Qwen3ForCausalLM` · **dense**

| | |
|---|---|
| Parameters | **8 B** total, 8 B active per token |
| Checkpoint on disk | 8.9 GB |
| Quantization | FP8 (block 128) |
| Layers / hidden | 36 / 4096 |
| Attn heads / KV heads | 32 / 8 |
| Vocab / max context | 151,936 / 40K |
| Tensor parallel | TP=1 |

Smallest tier and the only single-GPU run. Ungated on HF, so it proves the serving path without a token. GQA 32:8.

### Tier 2 — `Llama-3.1-70B-Instruct-FP8`

[`RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8`](https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8) · `LlamaForCausalLM` · **dense**

| | |
|---|---|
| Parameters | **70 B** total, 70 B active per token |
| Checkpoint on disk | 68 GB |
| Quantization | FP8 W8A8 (compressed-tensors) |
| Layers / hidden | 80 / 8192 |
| Attn heads / KV heads | 64 / 8 |
| Vocab / max context | 128,256 / 131K |
| Tensor parallel | TP=8 |

The dense headline. At TP=8 every layer all-reduces, so this is the tier that puts RCCL in the per-token critical path. RedHatAI quant chosen because meta-llama is gated.

### Are these numbers what the hardware should give?

**Roofline tok/s** = the most tokens/s possible if reading weights from HBM were the *only* cost. One decode step emits `batch` tokens and must read every weight it activates once, so:

```
step_time >= weight_bytes / (HBM_BW_per_GPU x GPUs)
roofline  =  batch / step_time
          =  batch x (HBM_BW_per_GPU x GPUs) / weight_bytes
```

e.g. Llama-70B: 256 x (8000 GB/s x 8) / 68 GB = **240,941 tok/s**.

`weight_bytes` is what is *actually read*, not model size — identical for dense models, but for Kimi-K3 it is **931 GB**, not 1.5 TB and not the ~84 B active params, because at batch 64 roughly 610 of 896 experts fire per layer. It deliberately ignores KV reads, compute, collectives and prefill, so it is a *loose upper bound*: far below it means weight traffic is not the constraint; near it means it is.

| Model | GPUs | Peak tok/s | tok/s **per GPU** | step (ms) | weights read/step | roofline tok/s | % of roofline |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-8B-FP8` | 1 | 14,988.0 | **14,988.0** | 17.1 | 8.0 GB | 256,000.0 | **5.9%** |
| `Llama-3.1-70B-Instruct-FP8` | 8 | 9,757.0 | **1,219.6** | 26.2 | 68.0 GB | 240,941.2 | **4.0%** |

**Yes — and the % column is the interesting part.** Kimi-K3 sits at ~29% of its weight-bandwidth ceiling while the two dense models sit at 4-6%. That is not Kimi doing better; it means Kimi is genuinely **bandwidth-bound** while Qwen and Llama are not. It also matches, independently, the ~29% HBM utilization measured in `kimi-k3-base.md` §3 — two different routes to the same number.

To be precise about *which* bandwidth: this is **intra-GPU HBM** — each GPU reading weights out of its own 8 TB/s on-package memory. It is **not** the XGMI GPU-to-GPU interconnect, which in the same run carries only activation all-reduces and sits at ~1% utilized. The two are often conflated; here they differ by roughly 390:1 in traffic. **See [`kimi-k3-base.md`](kimi-k3-base.md) for the full breakdown** — §3 ranks the three candidate bottlenecks (compute 1.1%, HBM ~29%, XGMI ~1.1%), §4 gives the per-step byte volumes on each path, and the terminology section at the top defines HBM vs XGMI.

#### If the dense models are not bandwidth-bound, what limits them?

**Not compute either.** Accounting for a Llama-70B decode step at c=256 (measured step time 27.4 ms):

| Candidate cost | Estimated per step | Share of step |
|---|---:|---:|
| Weight reads (68 GB / 8 GPUs at 8 TB/s) | ~1.1 ms | ~4% |
| KV-cache reads (~7.9 GB/GPU) | ~1.0 ms | ~4% |
| Decode compute (2 x 70e9 x 256 FLOP) | ~3.4 ms | ~12% |
| TP all-reduce (160 calls/step) | ~2-3 ms | ~10% |
| **Unaccounted** | **~19 ms** | **~70%** |

No single hardware resource is saturated: bandwidth ~4%, compute ~12%, interconnect ~10%. Calling these models "compute-bound" would be wrong. The ~70% residual is almost certainly **prefill interleaved with decode**, plus per-step scheduler overhead.

**What prefill and decode are.** Serving a request has two phases. *Prefill* is the one-time pass over the whole input prompt — all 1024 tokens processed in a single parallel forward pass, compute-heavy, once per request. *Decode* is everything after: output tokens generated one at a time, each a separate forward pass reading the KV cache. All the throughput and TPOT numbers here measure decode, and the roofline above models decode only.

**Why prefill becomes the limiter.** ATOM (like vLLM) uses continuous batching with chunked prefill: prefill and decode share the same GPU time slice, and a request cannot decode until it is prefilled. At c=256 with ISL=1024 that is **~262,000 prompt tokens** of backlog competing for the same GPU. While the scheduler runs prefill chunks, decode steps for in-flight requests wait. The signature is visible in the data: TTFT **median stays low (182 ms) while p99 blows out to 5,470 ms** — most requests prefill quickly, the tail queues behind the backlog.

**"Scheduling"** is the per-step cost of the serving loop itself — choosing which requests enter this step's batch, KV-cache block allocation, continuous-batching bookkeeping. It is CPU-side work on the critical path of every step, independent of GPU load.

**Why Kimi-K3 escapes this.** Its concurrency is capped at 64, so the prefill backlog is far smaller — and its per-step GPU work (931 GB of weight reads) is so large that prefill and scheduler overhead are comparatively negligible. That is much of why its roofline utilization (29%) looks so much healthier than the dense models' (4-6%): for Kimi the GPU-bound part genuinely dominates the step, while for Qwen and Llama the GPU-bound part is small and everything else fills the remaining ~70%.

So the three tiers are limited by three different things: **Kimi-K3 by memory bandwidth**, and **Qwen / Llama by prefill throughput and scheduling**, with no hardware unit near its ceiling. Their distance from the roofline is therefore expected rather than a defect — a *pure-decode* roofline is the wrong yardstick for a **mixed prefill+decode** serving benchmark.

> **This last part is inference, not measurement.** The residual is what is left after subtracting four estimated costs; it is not a profiled breakdown, and the individual estimates carry their own error. The TTFT median-vs-p99 gap is real measured evidence that queueing happens, but the *split* between prefill contention and scheduler overhead inside that ~70% is not measured — no trace shows "X ms prefill, Y ms scheduler". Settling it needs a profiler trace, or a decode-only run (short ISL, or prefill and decode measured separately).

### Reading the comparison

- **8B vs 70B — tracks model size, roughly.** Raw throughput differs only 1.54x, but that hides the GPU count: **per GPU** it is 14,988 vs 1,220 tok/s, a **12.3x** gap against an 8.8x active-parameter ratio. So the 70B recovers most of what its size costs by using 8 GPUs; the residual ~1.5x beyond pure size scaling is TP communication, a larger KV cache per token, and lower per-GPU efficiency. That is the expected shape.

> **Caveat: different TP.** Tier 1 is TP=1 (single GPU), tiers 2 and 3 are TP=8. Throughput is therefore not normalized per GPU, and the tiers answer "what can this box serve for this model" rather than "which model is more efficient per GPU".

## Tier 1 — `Qwen3-8B-FP8` (TP=1)

| Concurrency | req/s | output tok/s | total tok/s | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | completed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.19 | 183.1 | 353.2 | 25.3 | 29.9 | 5.44 | 5.46 | 10 |
| 2 | 0.37 | 335.8 | 678.5 | 29.3 | 41.6 | 5.81 | 5.84 | 20 |
| 4 | 0.71 | 664.0 | 1,320.4 | 29.9 | 62.5 | 5.92 | 5.95 | 40 |
| 8 | 1.38 | 1,280.1 | 2,558.6 | 30.0 | 119.0 | 6.07 | 6.14 | 80 |
| 16 | 2.66 | 2,457.5 | 4,918.2 | 30.1 | 198.5 | 6.35 | 6.43 | 160 |
| 32 | 4.78 | 4,428.4 | 8,850.7 | 30.4 | 414.8 | 6.91 | 7.06 | 320 |
| 64 | 8.39 | 7,737.9 | 15,509.2 | 31.2 | 834.9 | 7.98 | 8.30 | 640 |
| 128 | 12.51 | 11,557.3 | 23,118.4 | 39.6 | 1,583.1 | 10.77 | 11.40 | 1,280 |
| 256 | 16.30 | 14,988.0 | 30,041.0 | 50.1 | 3,083.1 | 16.76 | 17.78 | 2,560 |

- Peak **14,988.0 tok/s** at concurrency 256.
- **No knee in the sampled range** — still scaling at the highest concurrency tested; the ceiling is set by `max_num_seqs`, not by saturation.
- Across the sweep TPOT grows 3.1x (5.44 -> 16.76 ms) while throughput grows 81.8x — the batching trade-off for this model.

## Tier 2 — `Llama-3.1-70B-Instruct-FP8` (TP=8)

| Concurrency | req/s | output tok/s | total tok/s | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | completed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.14 | 135.2 | 260.8 | 104.3 | 113.1 | 7.30 | 7.34 | 10 |
| 2 | 0.26 | 236.7 | 478.3 | 101.4 | 122.6 | 8.20 | 8.32 | 20 |
| 4 | 0.51 | 479.1 | 952.6 | 100.5 | 179.4 | 8.14 | 8.38 | 40 |
| 8 | 0.97 | 895.3 | 1,789.5 | 64.8 | 206.7 | 8.65 | 8.99 | 80 |
| 16 | 1.79 | 1,651.8 | 3,305.9 | 52.3 | 397.8 | 9.36 | 9.94 | 160 |
| 32 | 3.14 | 2,906.3 | 5,808.6 | 53.1 | 738.5 | 10.49 | 11.31 | 320 |
| 64 | 5.22 | 4,810.2 | 9,641.3 | 55.2 | 1,347.4 | 12.90 | 13.72 | 640 |
| 128 | 7.94 | 7,328.8 | 14,660.0 | 79.3 | 2,564.1 | 17.05 | 18.64 | 1,280 |
| 256 | 10.61 | 9,757.0 | 19,556.3 | 96.0 | 4,733.7 | 25.89 | 28.03 | 2,560 |

- Peak **9,757.0 tok/s** at concurrency 256.
- **No knee in the sampled range** — still scaling at the highest concurrency tested; the ceiling is set by `max_num_seqs`, not by saturation.
- Across the sweep TPOT grows 3.5x (7.30 -> 25.89 ms) while throughput grows 72.2x — the batching trade-off for this model.

## Metric definitions

| Metric | Meaning |
|---|---|
| TTFT | Time to first token — prefill latency; the user-perceived lag. |
| TPOT | Time per output token — steady-state decode speed after the first token. |
| output tok/s | Generated tokens/s across all concurrent requests. |
| total tok/s | Input + output tokens/s (prefill work included). |

## Caveats

- **The load generator is co-located with the server**, competing for host CPU. Standard ATOM/vLLM practice, but not a clean client/server split; req/s at high concurrency is mildly pessimistic.
- `--ignore-eos` forces exactly OSL tokens per request, so throughput is not skewed by early stopping — comparable, but not representative of real traffic.
- `--random-range-ratio 0.8` jitters prompt lengths so prefix caching cannot inflate results.
- KV cache dtype is fp8; Kimi-K3 additionally runs with prefix caching disabled (required — KDA recurrent state cannot be rebuilt from the paged cache).

## Deep dive

`kimi-k3-base.md` analyses tier 3 in detail: achieved TFLOP/s, the GPU memory breakdown (weights vs KV pool), why the workload is HBM-bandwidth-bound rather than compute- or interconnect-bound, and the intra-GPU vs intra-node data volumes.

## Source data

| What | Where |
|---|---|
| Per-concurrency JSON / logs | `logs/atom/sweep_*/c<N>.{json,log}` |
| Sweep summaries | `logs/atom/sweep_*/summary.txt` |
| Server logs | `logs/atom/server_*/atom_server.log` |
| This table as CSV | `results/atom.csv` |

