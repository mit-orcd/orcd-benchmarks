# amd-ubuntu — ATOM LLM serving (Qwen3-8B, Llama-3.1-70B)

Generated 2026-10-09 06:00 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 for RVS (ROCm Validation Suite) and rccl-tests, containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **node6100 and node6101 agree within 1%** at every concurrency for both models.
- Llama-3.1-70B-FP8 (TP 8): ≈135 tok/s at 1 user, ≈9,700 tok/s at 256; TPOT (time per output token) 7.3 ms at 1 user.
- Qwen3-8B-FP8: ≈180 tok/s at 1 user, ≈15,000 tok/s at 256; TTFT (time to first token) stays ≈30 ms up to 64 users.
- Throughput rises smoothly with concurrency and TTFT/TPOT grow only at 128–256 users; no errors or outliers.

## Results

ATOM serving, ISL/OSL 1024/1024, image `atom-dev:nightly_202608111555`. Qwen3-8B-FP8 on 1 GPU (TP1), Llama-3.1-70B-FP8 on 8 GPUs (TP8). Kimi-K3 is in `kimi.md`.

### Output tok/s

| model | max_concurrency | node6100 | node6101 | 6100/6101 |
|---|---|---:|---:|---:|
| Llama-3.1-70B-Instruct-FP8 | 1 | 135 | 135 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 2 | 235 | 237 | 0.99x |
| Llama-3.1-70B-Instruct-FP8 | 4 | 479 | 479 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 8 | 893 | 895 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 16 | 1,647 | 1,652 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 32 | 2,882 | 2,906 | 0.99x |
| Llama-3.1-70B-Instruct-FP8 | 64 | 4,817 | 4,810 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 128 | 7,293 | 7,329 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 256 | 9,639 | 9,757 | 0.99x |
| Qwen3-8B-FP8 | 1 | 180 | 183 | 0.99x |
| Qwen3-8B-FP8 | 2 | 335 | 336 | 1.00x |
| Qwen3-8B-FP8 | 4 | 665 | 664 | 1.00x |
| Qwen3-8B-FP8 | 8 | 1,283 | 1,280 | 1.00x |
| Qwen3-8B-FP8 | 16 | 2,454 | 2,457 | 1.00x |
| Qwen3-8B-FP8 | 32 | 4,430 | 4,428 | 1.00x |
| Qwen3-8B-FP8 | 64 | 7,749 | 7,738 | 1.00x |
| Qwen3-8B-FP8 | 128 | 11,546 | 11,557 | 1.00x |
| Qwen3-8B-FP8 | 256 | 14,967 | 14,988 | 1.00x |

### Median TTFT (ms)

| model | max_concurrency | node6100 | node6101 | 6100/6101 |
|---|---|---:|---:|---:|
| Llama-3.1-70B-Instruct-FP8 | 1 | 106.3 | 104.3 | 1.02x |
| Llama-3.1-70B-Instruct-FP8 | 2 | 104.3 | 101.4 | 1.03x |
| Llama-3.1-70B-Instruct-FP8 | 4 | 104.2 | 100.5 | 1.04x |
| Llama-3.1-70B-Instruct-FP8 | 8 | 63.4 | 64.8 | 0.98x |
| Llama-3.1-70B-Instruct-FP8 | 16 | 52.2 | 52.3 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 32 | 53.4 | 53.1 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 64 | 54.9 | 55.2 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 128 | 76.8 | 79.3 | 0.97x |
| Llama-3.1-70B-Instruct-FP8 | 256 | 95.1 | 96.0 | 0.99x |
| Qwen3-8B-FP8 | 1 | 25.0 | 25.3 | 0.99x |
| Qwen3-8B-FP8 | 2 | 29.2 | 29.3 | 1.00x |
| Qwen3-8B-FP8 | 4 | 29.7 | 29.9 | 0.99x |
| Qwen3-8B-FP8 | 8 | 29.8 | 30.0 | 0.99x |
| Qwen3-8B-FP8 | 16 | 29.9 | 30.1 | 0.99x |
| Qwen3-8B-FP8 | 32 | 30.1 | 30.4 | 0.99x |
| Qwen3-8B-FP8 | 64 | 30.7 | 31.2 | 0.98x |
| Qwen3-8B-FP8 | 128 | 39.5 | 39.6 | 1.00x |
| Qwen3-8B-FP8 | 256 | 49.0 | 50.1 | 0.98x |

### Median TPOT (ms)

| model | max_concurrency | node6100 | node6101 | 6100/6101 |
|---|---|---:|---:|---:|
| Llama-3.1-70B-Instruct-FP8 | 1 | 7.31 | 7.30 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 2 | 8.25 | 8.20 | 1.01x |
| Llama-3.1-70B-Instruct-FP8 | 4 | 8.15 | 8.14 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 8 | 8.66 | 8.65 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 16 | 9.39 | 9.36 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 32 | 10.59 | 10.49 | 1.01x |
| Llama-3.1-70B-Instruct-FP8 | 64 | 12.88 | 12.90 | 1.00x |
| Llama-3.1-70B-Instruct-FP8 | 128 | 17.15 | 17.05 | 1.01x |
| Llama-3.1-70B-Instruct-FP8 | 256 | 26.26 | 25.89 | 1.01x |
| Qwen3-8B-FP8 | 1 | 5.52 | 5.44 | 1.02x |
| Qwen3-8B-FP8 | 2 | 5.82 | 5.81 | 1.00x |
| Qwen3-8B-FP8 | 4 | 5.91 | 5.92 | 1.00x |
| Qwen3-8B-FP8 | 8 | 6.06 | 6.07 | 1.00x |
| Qwen3-8B-FP8 | 16 | 6.36 | 6.35 | 1.00x |
| Qwen3-8B-FP8 | 32 | 6.90 | 6.91 | 1.00x |
| Qwen3-8B-FP8 | 64 | 7.97 | 7.98 | 1.00x |
| Qwen3-8B-FP8 | 128 | 10.78 | 10.77 | 1.00x |
| Qwen3-8B-FP8 | 256 | 16.77 | 16.76 | 1.00x |

