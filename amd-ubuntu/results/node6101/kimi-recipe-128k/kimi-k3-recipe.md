# Kimi-K3 — vLLM recipe 1 (recipes.vllm.ai, MI355X, 2026-09-25)

Image: `vllm/vllm-openai-rocm:nightly-rocm100 -> sha256:e76a953fa2c317e5e0913fab946887ae498629c8e57fb6b2dc633dadd4098b61`. TP8 on one node, MXFP4 experts, FP8 KV cache, prefix caching, one server per concurrency point with the recipe's per-point settings (DSpark speculative decoding up to C=14, decode-context-parallel 8 + CPU KV offload above). Workload: random ISL/OSL 1024/1024, `--ignore-eos`, 10×C prompts, `vllm bench serve`.
Raw logs: `kimi_recipe_20261008_220520`.

TPOT = time per output token.

TTFT = time to first token.

| C | draft K | max-num-seqs | DCP | KV offload | out tok/s | tok/s per user | TTFT med (ms) | TPOT med (ms) | TPOT p99 (ms) | accept len | completed |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 2 | 1 | no | — | — | — | — | — | 1.13 | — |
| 2 | 5 | 4 | 1 | no | — | — | — | — | — | 1.00 | — |
| 4 | 5 | 8 | 1 | no | — | — | — | — | — | 1.25 | — |
| 8 | 4 | 16 | 1 | yes | — | — | — | — | — | 1.21 | — |
| 16 | 0 | 32 | 8 | yes | — | — | — | — | — | — | — |
| 32 | 0 | 64 | 8 | yes | — | — | — | — | — | — | — |
| 64 | 0 | 128 | 8 | yes | — | — | — | — | — | — | — |
| 128 | 0 | 256 | 8 | yes | — | — | — | — | — | — | — |

No result at C = 1, 2, 4, 8, 16, 32, 64, 128: see `c<C>/server.log` and `bench.log`.

tok/s per user = 1000 / median TPOT. With speculative decoding TPOT counts accepted tokens, so it already includes the draft's benefit.
