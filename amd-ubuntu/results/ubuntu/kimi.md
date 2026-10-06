# amd-ubuntu — Kimi-K3 serving: ATOM and the AMD vLLM recipe

Generated 2026-10-06 00:30 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 (RVS, rccl-tests), containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **Model weights**: complete copy at `/scratch/Kimi-K3` on both nodes (verified before the runs).
- **ATOM** (same images as amd-cloud): ≈45 tok/s at 1 user up to ≈2,600 tok/s at 256 (best config `maxseqs`).
- **AMD vLLM recipe** (DSpark speculative decoding up to C=14, DCP 8 + CPU KV offload above): **1.56× ATOM at 1 user, 1.39× at 4, 1.24× at 8**, with 19–29% lower TPOT; 1.05× at 64, about equal at 128–256 (0.97× at 256). The recipe is the better choice for latency-sensitive, low-concurrency serving; ATOM is equal or slightly better at high concurrency.
- `max-num-seqs 2048` does not fit: each request needs ≈107 GB of cache against a 58 GB KV budget. Not a node problem; the configuration is too large for 8 × MI355X.

## Results

Kimi-K3 on 8 × MI355X, TP8, weights from node-local `/scratch/Kimi-K3`, ISL/OSL 1024/1024 unless noted. Two stacks:

- **ATOM** (`kimi-cloud`, node6100): `atom-dev:nightly_202608111555` (most experiments), `nightly_202608191459` (isl4096, repeats arm A), MAD `rocm7.2.4_..._20260727_kimi_k3` (mad, single_stream, repeats arm B, ep_matched).
- **vLLM recipe** (`kimi-recipe`, node6101): AMD's vLLM recipe (recipes.vllm.ai, 2026-09-25): `vllm/vllm-openai-rocm:nightly-rocm100` (digest e76a953f), DSpark speculative decoding up to C=14, decode-context-parallel 8 + CPU KV offload above, server re-tuned per concurrency.

### What improved: vLLM recipe vs best ATOM result at the same concurrency

| C | recipe tok/s | ATOM best tok/s (sweep) | tok/s ratio | recipe TPOT ms | ATOM TPOT ms | TPOT ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71 | 45 (base) | **1.56x** | 15.36 | 21.76 | **0.71x** |
| 4 | 212 | 153 (base) | **1.39x** | 18.79 | 25.23 | **0.74x** |
| 8 | 353 | 285 (base) | **1.24x** | 22.00 | 27.26 | **0.81x** |
| 10 | 419 | — | — | 23.25 | — | — |
| 12 | 495 | — | — | 23.32 | — | — |
| 14 | 549 | — | — | 25.23 | — | — |
| 44 | 1,113 | — | — | 38.26 | — | — |
| 48 | 1,158 | — | — | 39.90 | — | — |
| 64 | 1,344 | 1,282 (base) | 1.05x | 45.56 | 49.05 | **0.93x** |
| 70 | 1,369 | — | — | 48.84 | — | — |
| 128 | 1,865 | 1,845 (maxseqs) | 1.01x | 66.90 | 68.87 | 0.97x |
| 256 | 2,522 | 2,602 (maxseqs) | 0.97x | 100.43 | 98.49 | 1.02x |

tok/s ratio above 1 and TPOT ratio below 1 favour the recipe; bold = more than 5%. ATOM has no run at C = 10, 12, 14, 44, 48, 70. The recipe also changes the stack (vLLM, ROCm 10.0 userspace), so this measures recipe and stack together.

### vLLM recipe sweep

Detail: `node6101/kimi-recipe/kimi-k3-recipe.md`.

| C | draft K | DCP | out tok/s | TPOT med ms | TTFT med ms | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 1 | 71 | 15.36 | 199 | 10 |
| 4 | 5 | 1 | 212 | 18.79 | 275 | 40 |
| 8 | 4 | 1 | 353 | 22.00 | 283 | 80 |
| 10 | 4 | 1 | 419 | 23.25 | 284 | 100 |
| 12 | 3 | 1 | 495 | 23.32 | 286 | 120 |
| 14 | 3 | 1 | 549 | 25.23 | 297 | 140 |
| 44 | 0 | 8 | 1,113 | 38.26 | 276 | 440 |
| 48 | 0 | 8 | 1,158 | 39.90 | 281 | 480 |
| 64 | 0 | 8 | 1,344 | 45.56 | 289 | 640 |
| 70 | 0 | 8 | 1,369 | 48.84 | 289 | 700 |
| 128 | 0 | 8 | 1,865 | 66.90 | 314 | 1280 |
| 256 | 0 | 8 | 2,522 | 100.43 | 480 | 2560 |

### ATOM experiments

#### base: ATOM recipe, max-num-seqs 64

| max_concurrency | tok/s | TPOT ms |
|---|---:|---:|
| 1 | 45.5 | 21.8 |
| 2 | 86.7 | 22.8 |
| 4 | 152.9 | 25.2 |
| 8 | 285.4 | 27.3 |
| 16 | 506.2 | 30.8 |
| 32 | 830.8 | 37.5 |
| 64 | 1,281.7 | 49.1 |

#### maxseqs: max-num-seqs 256

| conc | tok/s | TPOT ms |
|---|---:|---:|
| 64 | 1,274.7 | 49.0 |
| 128 | 1,844.7 | 68.9 |
| 256 | 2,601.5 | 98.5 |

#### mad: MAD recipe, max-num-seqs 64

| conc | tok/s | TPOT ms |
|---|---:|---:|
| 64 | 1,139.3 | 52.6 |
| 128 | 1,206.8 | 52.4 |
| 256 | 1,199.9 | 53.1 |

#### max-num-seqs 512

| conc | tok/s | TPOT ms |
|---|---:|---:|
| 64 | 1,265.9 | 48.9 |
| 128 | 1,827.1 | 68.7 |
| 256 | 2,547.3 | 99.2 |
| 512 | 3,561.6 | 143.5 |

#### max-num-seqs 1024

| conc | tok/s | TPOT ms |
|---|---:|---:|
| 256 | 2,559.9 | 99.1 |
| 512 | 2,235.0 | 133.7 |
| 1024 | 2,222.2 | 134.1 |

#### max-num-seqs 2048

*Not runnable on 8 × MI355X at TP8, on amd-cloud (2026-08-20) and amd-ubuntu (2026-10-02, 2026-10-04) alike.* ATOM reserves Kimi-K3's KDA recurrent state (FP32) per sequence slot before the paged KV cache: 107 GiB per GPU for 2048 slots, against ~58 GiB left after the 190 GiB of weights and activations at `--gpu-memory-utilization 0.93` (ATOM: "would need 1.10"). max-num-seqs 1024 (54 GiB of state) is the largest power of two that fits.

#### isl4096: ISL 4096 / OSL 1024

| conc | tok/s | TTFT ms |
|---|---:|---:|
| 64 | 1,206.0 | 319.7 |
| 128 | 1,674.9 | 468.5 |
| 256 | 2,114.6 | 556.2 |

#### single_stream: latency arms

| arm | concurrency | TPOT ms | tok/s |
|---|---|---:|---:|
| K1_mad_default | 1 | 24.7 | 40.2 |
| K1_mad_default | 2 | 25.4 | 76.8 |
| K1_mad_default | 4 | 26.7 | 143.8 |
| K1_mad_default | 8 | 28.8 | 263.2 |
| K3_aiter_attn | 1 | 24.5 | 40.4 |
| K3_aiter_attn | 2 | 25.2 | 77.2 |
| K3_aiter_attn | 4 | 26.6 | 144.1 |
| K3_aiter_attn | 8 | 28.9 | 263.4 |
| K4_grouped_gemm | 1 | 23.8 | 41.6 |
| K4_grouped_gemm | 2 | 24.6 | 79.1 |
| K4_grouped_gemm | 4 | 26.0 | 147.7 |
| K4_grouped_gemm | 8 | 28.1 | 270.6 |

#### repeats: run-to-run spread (tok/s, c=256)

| config | mean | std | n |
|---|---:|---:|---:|
| A_original | 1,404 | 15 | 3 |
| B_mad | 1,212 | 1 | 3 |

Per-experiment detail (server logs, profiler, EP): `node6100/kimi-cloud/kimi-k3-*.md`.
