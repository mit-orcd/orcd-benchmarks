# ATOM serving benchmark — MI355X

Engine: [ATOM](https://github.com/ROCm/ATOM) (AITER-optimized, vLLM-like) on 8 x MI355X (gfx950), ROCm 7.2.4. Workload: ISL/OSL 1024/1024, `--ignore-eos`, saturating request rate.

Unlike Parts A-C this measures **inference serving**, not raw FLOPS or fabric bandwidth. There is no `dell-cloud/` baseline for this suite — compare against ATOM's public dashboard, not against this repo.

## Summary — three models at a glance

| Tier | Model | Params (total / active) | On disk | TP | Peak tok/s | @ conc | TTFT med @c=1 (ms) | TPOT med @c=1 (ms) | Knee |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tier 3 | `Kimi-K3` | 2.78 T / ~84 B | 1.5 TB | 8 | **1,281.7** | 64 | 225.0 | 21.76 | none in range |

*Active* params are what actually fire per token — identical to total for a dense model, but only ~3% of total for Kimi-K3's MoE. That distinction drives most of the throughput differences below.

## The three models

### Tier 3 — `Kimi-K3`

[`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3) · `KimiK3ForConditionalGeneration` · **MoE (hybrid attn)**

| | |
|---|---|
| Parameters | **2.78 T** total, ~84 B active per token |
| Checkpoint on disk | 1.5 TB |
| Quantization | MXFP4 experts + PTPC-FP8 rest |
| Layers / hidden | 93 / 7168 |
| Attn heads / KV heads | 96 / 96 |
| Vocab / max context | 163,840 / 1M |
| Tensor parallel | TP=8 |

Frontier MoE: 896 routed experts, top-16 + 2 shared, so only ~3% of the model fires per token. 24 MLA full-attention layers + 69 KDA linear-attention layers — only the 24 keep a growing KV cache.

## Tier 3 — `Kimi-K3` (TP=8)

| Concurrency | req/s | output tok/s | total tok/s | TTFT med (ms) | TTFT p99 (ms) | TPOT med (ms) | TPOT p99 (ms) | completed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 45.5 | 90.6 | 225.0 | 294.5 | 21.76 | 21.84 | 10 |
| 2 | 0.09 | 86.7 | 171.8 | 250.1 | 255.7 | 22.80 | 22.97 | 20 |
| 4 | 0.17 | 152.9 | 307.2 | 253.8 | 380.7 | 25.23 | 25.50 | 40 |
| 8 | 0.31 | 285.4 | 568.8 | 255.7 | 448.2 | 27.26 | 28.12 | 80 |
| 16 | 0.55 | 506.2 | 1,017.7 | 251.6 | 990.2 | 30.79 | 31.80 | 160 |
| 32 | 0.90 | 830.8 | 1,658.9 | 261.2 | 2,494.6 | 37.51 | 39.21 | 320 |
| 64 | 1.39 | 1,281.7 | 2,564.0 | 274.4 | 3,534.5 | 49.05 | 52.00 | 640 |

- Peak **1,281.7 tok/s** at concurrency 64.
- **No knee in the sampled range** — still scaling at the highest concurrency tested; the ceiling is set by `max_num_seqs`, not by saturation.
- Across the sweep TPOT grows 2.3x (21.76 -> 49.05 ms) while throughput grows 28.2x — the batching trade-off for this model.

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

