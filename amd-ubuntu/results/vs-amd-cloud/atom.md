# ATOM serving: amd-ubuntu vs amd-cloud

Generated 2026-10-07 21:04 by `report.py`.

- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker ([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).
- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the containers) the same image digests.
- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT, TPOT) below 1 is better.


## Analysis

- **Same images and digests; amd-ubuntu is faster than amd-cloud at every concurrency below 256.**
  - Llama-3.1-70B-FP8: output throughput 1.14–1.34× (largest at 1–8 users), TPOT 0.74–0.88×, TTFT 0.60–0.90× from 8 users up.
  - Qwen3-8B-FP8: 1.07–1.16× throughput, TPOT 0.87–0.95×, TTFT 0.55–0.80× (0.27× at 256).
- At 256 users both systems are equal (1.00–1.04×), where the GPUs are compute-bound.
- The gain is largest where per-request overhead (CPU, launch latency) matters most, which points at host-side differences (newer OS/kernel, CPU settings) rather than the GPUs.
- One exception: Llama-70B TTFT at 1–4 users is ≈1.45× slower on amd-ubuntu (≈104 vs 72 ms). Minor in absolute terms; not investigated.
- Both amd-ubuntu nodes agree within 1%, so the result is reproducible.

## Results

Same image digest on both systems (`atom-dev:nightly_202608111555`, amd-cloud's `:latest` of 2026-08-14). Kimi-K3 is in `kimi.md`.

### Output tok/s

| model | max_concurrency | amd-cloud | node6100 | node6101 | 6100/cloud | 6101/cloud |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-70B-Instruct-FP8 | 1 | 101 | 135 | 135 | **1.34x** | **1.34x** |
| Llama-3.1-70B-Instruct-FP8 | 2 | 191 | 235 | 237 | **1.23x** | **1.24x** |
| Llama-3.1-70B-Instruct-FP8 | 4 | 374 | 479 | 479 | **1.28x** | **1.28x** |
| Llama-3.1-70B-Instruct-FP8 | 8 | 711 | 893 | 895 | **1.26x** | **1.26x** |
| Llama-3.1-70B-Instruct-FP8 | 16 | 1,352 | 1,647 | 1,652 | **1.22x** | **1.22x** |
| Llama-3.1-70B-Instruct-FP8 | 32 | 2,432 | 2,882 | 2,906 | **1.18x** | **1.19x** |
| Llama-3.1-70B-Instruct-FP8 | 64 | 4,125 | 4,817 | 4,810 | **1.17x** | **1.17x** |
| Llama-3.1-70B-Instruct-FP8 | 128 | 6,413 | 7,293 | 7,329 | **1.14x** | **1.14x** |
| Llama-3.1-70B-Instruct-FP8 | 256 | 9,342 | 9,639 | 9,757 | 1.03x | 1.04x |
| Qwen3-8B-FP8 | 1 | 159 | 180 | 183 | **1.13x** | **1.15x** |
| Qwen3-8B-FP8 | 2 | 301 | 335 | 336 | **1.11x** | **1.12x** |
| Qwen3-8B-FP8 | 4 | 574 | 665 | 664 | **1.16x** | **1.16x** |
| Qwen3-8B-FP8 | 8 | 1,146 | 1,283 | 1,280 | **1.12x** | **1.12x** |
| Qwen3-8B-FP8 | 16 | 2,183 | 2,454 | 2,457 | **1.12x** | **1.13x** |
| Qwen3-8B-FP8 | 32 | 4,029 | 4,430 | 4,428 | **1.10x** | **1.10x** |
| Qwen3-8B-FP8 | 64 | 7,259 | 7,749 | 7,738 | **1.07x** | **1.07x** |
| Qwen3-8B-FP8 | 128 | 11,111 | 11,546 | 11,557 | 1.04x | 1.04x |
| Qwen3-8B-FP8 | 256 | 14,963 | 14,967 | 14,988 | 1.00x | 1.00x |

### Median TTFT (ms)

| model | max_concurrency | amd-cloud | node6100 | node6101 | 6100/cloud | 6101/cloud |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-70B-Instruct-FP8 | 1 | 71.9 | 106.3 | 104.3 | **1.48x** | **1.45x** |
| Llama-3.1-70B-Instruct-FP8 | 2 | 71.2 | 104.3 | 101.4 | **1.46x** | **1.42x** |
| Llama-3.1-70B-Instruct-FP8 | 4 | 72.1 | 104.2 | 100.5 | **1.45x** | **1.39x** |
| Llama-3.1-70B-Instruct-FP8 | 8 | 71.9 | 63.4 | 64.8 | **0.88x** | **0.90x** |
| Llama-3.1-70B-Instruct-FP8 | 16 | 71.8 | 52.2 | 52.3 | **0.73x** | **0.73x** |
| Llama-3.1-70B-Instruct-FP8 | 32 | 72.8 | 53.4 | 53.1 | **0.73x** | **0.73x** |
| Llama-3.1-70B-Instruct-FP8 | 64 | 91.9 | 54.9 | 55.2 | **0.60x** | **0.60x** |
| Llama-3.1-70B-Instruct-FP8 | 128 | 110.3 | 76.8 | 79.3 | **0.70x** | **0.72x** |
| Llama-3.1-70B-Instruct-FP8 | 256 | 154.5 | 95.1 | 96.0 | **0.62x** | **0.62x** |
| Qwen3-8B-FP8 | 1 | 45.8 | 25.0 | 25.3 | **0.55x** | **0.55x** |
| Qwen3-8B-FP8 | 2 | 36.5 | 29.2 | 29.3 | **0.80x** | **0.80x** |
| Qwen3-8B-FP8 | 4 | 42.7 | 29.7 | 29.9 | **0.70x** | **0.70x** |
| Qwen3-8B-FP8 | 8 | 42.3 | 29.8 | 30.0 | **0.71x** | **0.71x** |
| Qwen3-8B-FP8 | 16 | 43.1 | 29.9 | 30.1 | **0.69x** | **0.70x** |
| Qwen3-8B-FP8 | 32 | 43.1 | 30.1 | 30.4 | **0.70x** | **0.70x** |
| Qwen3-8B-FP8 | 64 | 53.3 | 30.7 | 31.2 | **0.58x** | **0.59x** |
| Qwen3-8B-FP8 | 128 | 68.2 | 39.5 | 39.6 | **0.58x** | **0.58x** |
| Qwen3-8B-FP8 | 256 | 182.2 | 49.0 | 50.1 | **0.27x** | **0.28x** |

### Median TPOT (ms)

| model | max_concurrency | amd-cloud | node6100 | node6101 | 6100/cloud | 6101/cloud |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-70B-Instruct-FP8 | 1 | 9.81 | 7.31 | 7.30 | **0.74x** | **0.74x** |
| Llama-3.1-70B-Instruct-FP8 | 2 | 10.36 | 8.25 | 8.20 | **0.80x** | **0.79x** |
| Llama-3.1-70B-Instruct-FP8 | 4 | 10.28 | 8.15 | 8.14 | **0.79x** | **0.79x** |
| Llama-3.1-70B-Instruct-FP8 | 8 | 10.93 | 8.66 | 8.65 | **0.79x** | **0.79x** |
| Llama-3.1-70B-Instruct-FP8 | 16 | 11.49 | 9.39 | 9.36 | **0.82x** | **0.81x** |
| Llama-3.1-70B-Instruct-FP8 | 32 | 12.74 | 10.59 | 10.49 | **0.83x** | **0.82x** |
| Llama-3.1-70B-Instruct-FP8 | 64 | 15.00 | 12.88 | 12.90 | **0.86x** | **0.86x** |
| Llama-3.1-70B-Instruct-FP8 | 128 | 19.47 | 17.15 | 17.05 | **0.88x** | **0.88x** |
| Llama-3.1-70B-Instruct-FP8 | 256 | 26.74 | 26.26 | 25.89 | 0.98x | 0.97x |
| Qwen3-8B-FP8 | 1 | 6.24 | 5.52 | 5.44 | **0.88x** | **0.87x** |
| Qwen3-8B-FP8 | 2 | 6.57 | 5.82 | 5.81 | **0.89x** | **0.88x** |
| Qwen3-8B-FP8 | 4 | 6.71 | 5.91 | 5.92 | **0.88x** | **0.88x** |
| Qwen3-8B-FP8 | 8 | 6.77 | 6.06 | 6.07 | **0.89x** | **0.90x** |
| Qwen3-8B-FP8 | 16 | 7.11 | 6.36 | 6.35 | **0.90x** | **0.89x** |
| Qwen3-8B-FP8 | 32 | 7.67 | 6.90 | 6.91 | **0.90x** | **0.90x** |
| Qwen3-8B-FP8 | 64 | 8.41 | 7.97 | 7.98 | **0.95x** | **0.95x** |
| Qwen3-8B-FP8 | 128 | 11.14 | 10.78 | 10.77 | 0.97x | 0.97x |
| Qwen3-8B-FP8 | 256 | 16.34 | 16.77 | 16.76 | 1.03x | 1.03x |

