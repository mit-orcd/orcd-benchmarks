# Kimi-K3 on 8 × MI355X: ATOM vs vLLM recipe 1 vs vLLM recipe 2

What is the same and what differs between our three Kimi-K3 serving setups on one MI355X node.
Results: [ubuntu/kimi-recipe-old-vs-new.md](ubuntu/kimi-recipe-old-vs-new.md), [ubuntu/kimi-atom-mtp.md](ubuntu/kimi-atom-mtp.md),
[mi355x-vs-b200.md](mi355x-vs-b200.md) §6. TPOT = time per output token, TTFT = time to first token.
Per-user tok/s = 1000 / median TPOT.

## Same in all three

| Item | Setting |
|---|---|
| Hardware | One node, 8 × MI355X (288 GB each), node6100 or node6101 |
| Parallelism | TP8, no expert parallelism, no pipeline parallelism |
| Model | `moonshotai/Kimi-K3`, native MXFP4 checkpoint, `/scratch/Kimi-K3` |
| KV cache | fp8 |
| Workload | random prompts, ISL/OSL 1024/1024, `--ignore-eos`, range ratio 0.8, 10 × users prompts |
| Containers | apptainer through the docker shim (`common/bin/docker`) |

## What differs

| Item | ATOM | vLLM recipe 1 | vLLM recipe 2 |
|---|---|---|---|
| Engine | AMD ATOM (not vLLM) | vLLM (ROCm) | vLLM (ROCm) |
| Image | `rocm/atom-dev:nightly_202608111555` (same as amd-cloud); MTP run: `nightly_202610011450` | `vllm/vllm-openai-rocm:nightly-rocm100` | `vllm/vllm-openai-rocm:v0.29.0` (stock) |
| Source | ROCm/MAD Kimi-K3 benchmark | recipes.vllm.ai, MI355X (2026-09-25) | AMD's `amd-kimi-k3-recipe.pdf` (2026-10) |
| Script | `atom/run_atom_server.sh` (part D), `atom/run_kimi_atom_mtp.sh` | `atom/run_kimi_recipe.sh` | `atom/run_kimi_amdrecipe.sh` |
| One server for the whole sweep? | Yes (one per `max-num-seqs` variant) | **No: restarted with new settings for each load** | Yes |
| `max-num-seqs` | 64 (variants 256, 512 for heavy load) | 2 × users | 128 |
| `max-num-batched-tokens` | 16384 | 16384 at 1 user, 8192 above | 4096 |
| `max-model-len` | 16384 | 1,048,576 (native 1M) | not set (model default) |
| Prefix caching | off | on | not set (vLLM default) |
| `gpu-memory-utilization` | 0.93 | 0.90 | 0.95 |
| Speculative decoding (MTP, DSpark draft) | off in the baseline; MTP run: draft length 6 at 1 user, 2 above | **on up to 14 users**: draft length 7 (1 user), 5 (2–4), 4 (5–10), 3 (11–14); off above | off |
| MTP acceptance rule | ATOM default (not set) | `draft_sample_method` probabilistic, `rejection_sample_method` block | – |
| Decode context parallel (DCP) | no | DCP 8 above 14 users | no |
| CPU KV-cache offload | no | from 8 users (`SimpleCPUOffloadConnector`) | no |
| CUDA graphs | ATOM default | full + piecewise up to 4 users, full above; every batch size captured | `FULL_DECODE_ONLY` |
| Quantization extras | online PTPC-FP8 for the non-expert layers | none (checkpoint as is) | none (checkpoint as is) |
| Kernels | ATOM's own (AITER) | AITER (`VLLM_ROCM_USE_AITER=1`, AITER MLA, INT4 quick all-reduce) | AITER (`VLLM_ROCM_USE_AITER=1`, A8W4 MoE) |
| Benchmark client | `atom.benchmarks.benchmark_serving` | `vllm bench serve` | `vllm bench serve` |
| Sampling (temperature) | not set (client/server default) | not set (client/server default) | not set (client/server default) |

## Results side by side (ISL/OSL 1024/1024)

| Users | | ATOM (base image) | ATOM, no MTP (new image) | ATOM +MTP (new image) | vLLM recipe 1 | vLLM recipe 2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | per-user tok/s | 46.0 | 64.0 | **256.8** | 65.1 (MTP on) | 55.1 |
| 1 | throughput tok/s | 45 | 63 | 197 | 71 | 54 |
| 8 | per-user tok/s | 36.7 | 56.4 | **112.2** | 45.5 (MTP on) | 41.9 |
| 8 | throughput tok/s | 285 | 431 | 812 | 353 | 315 |
| 64 | throughput tok/s | 1,282 | – | **2,463** | 1,344 (no MTP) | 1,228 |
| 128 | throughput tok/s | 1,845 (`max-num-seqs` 256) | – | – | **1,865** | 1,735 |
| 256 | throughput tok/s | **2,602** (`max-num-seqs` 256) | – | – | 2,522 | 1,747 |

Base image = `nightly_202608111555`, new image = `nightly_202610011450`. The new ATOM image alone is 1.4× faster
at 1 user (64.0 vs 46.0 per-user tok/s, no MTP in either).

## MTP: same draft model, very different gain

| | ATOM +MTP | vLLM recipe 1 |
|---|---|---|
| Draft model | `Inferact/Kimi-K3-DSpark` | `Inferact/Kimi-K3-DSpark` |
| Draft length | 6 (1 user), 2 (2+ users) | 7, 5, 4, 3 (1–14 users) |
| Tokens per step (measured) | ≈4.3 at draft 6; ≈2.7 of max 3 at draft 2 | 1.30–1.66 |
| Gain at 1 user | 4.0× (64.0 → 256.8 per-user tok/s, same image) | small: 65.1 with MTP vs 64.0 for ATOM without it |

The same draft on the same prompts should not give a 3–5× difference in acceptance. Possible causes, not yet
checked: the vLLM draft setup (on B200 the draft needed a `pard_token` config fix), different default
sampling or acceptance rules between the engines, or ATOM accepting too much. A greedy check (MTP output
must equal no-MTP output) would tell. Random prompts with `--ignore-eos` also make text easier to guess,
so both engines' MTP gains are likely higher here than on real text.

## Which to use (from these runs)

| Load | Fastest | Note |
|---|---|---|
| 1–64 users | ATOM +MTP (new image) | if its MTP output checks out (see above) |
| 128 users | vLLM recipe 1 | ATOM +MTP not run at 128; recipe 1 re-tunes the server for each load |
| 2–32 users, one fixed setting | vLLM recipe 2 | simplest; raise `max-num-seqs` above 128 for heavier load |
| 256+ users | ATOM, `max-num-seqs` 256–512 | ATOM +MTP not run above 64 |
