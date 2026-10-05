# Kimi-K3 — AMD vLLM recipe (recipes.vllm.ai, MI355X, 2026-09-25)

Image: `vllm/vllm-openai-rocm:nightly-rocm100 -> sha256:e76a953fa2c317e5e0913fab946887ae498629c8e57fb6b2dc633dadd4098b61`. TP8 on one node, MXFP4 experts, FP8 KV cache, prefix caching, one server per concurrency point with the recipe's per-point settings (DSpark speculative decoding up to C=14, decode-context-parallel 8 + CPU KV offload above). Workload: random ISL/OSL 1024/1024, `--ignore-eos`, 10×C prompts, `vllm bench serve`.
Raw logs: `kimi_recipe_20261002_024747, kimi_recipe_20261005_001645, kimi_recipe_20261005_030557`.

| C | draft K | max-num-seqs | DCP | KV offload | out tok/s | tok/s per user | TTFT med (ms) | TPOT med (ms) | TPOT p99 (ms) | accept len | completed |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 2 | 1 | no | 71 | 65.1 | 199 | 15.36 | 15.79 | 1.30 | 10 |
| 4 | 5 | 8 | 1 | no | 212 | 53.2 | 275 | 18.79 | 21.20 | 1.39 | 40 |
| 8 | 4 | 16 | 1 | yes | 353 | 45.5 | 283 | 22.00 | 26.83 | 1.41 | 80 |
| 10 | 4 | 20 | 1 | yes | 419 | 43.0 | 284 | 23.25 | 26.29 | 1.66 | 100 |
| 12 | 3 | 24 | 1 | yes | 495 | 42.9 | 286 | 23.32 | 26.42 | 1.50 | 120 |
| 14 | 3 | 28 | 1 | yes | 549 | 39.6 | 297 | 25.23 | 28.68 | 1.45 | 140 |
| 44 | 0 | 88 | 8 | yes | 1,113 | 26.1 | 276 | 38.26 | 46.10 | — | 440 |
| 48 | 0 | 96 | 8 | yes | 1,158 | 25.1 | 281 | 39.90 | 44.45 | — | 480 |
| 64 | 0 | 128 | 8 | yes | 1,344 | 21.9 | 289 | 45.56 | 49.50 | — | 640 |
| 70 | 0 | 140 | 8 | yes | 1,369 | 20.5 | 289 | 48.84 | 61.40 | — | 700 |
| 128 | 0 | 256 | 8 | yes | 1,865 | 14.9 | 314 | 66.90 | 72.78 | — | 1280 |
| 256 | 0 | 512 | 8 | yes | 2,522 | 10.0 | 480 | 100.43 | 107.50 | — | 2560 |

tok/s per user = 1000 / median TPOT. With speculative decoding TPOT counts accepted tokens, so it already includes the draft's benefit.
