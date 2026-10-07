# Kimi-K3: amd-ubuntu vs amd-cloud

Generated 2026-10-07 20:01 by `report.py`.

- **amd-cloud**: one 8 × MI355X node, ROCm 7.14, Ubuntu 22.04.5, docker ([results](../../../amd-benchmarks/amd-cloud/results/SUMMARY.md)).
- **amd-ubuntu**: node6100 and node6101, 8 × MI355X each, same amdgpu driver and CPUs, host ROCm 7.2.4, Ubuntu 24.04.5, apptainer. Same scripts, analyzers and (for the containers) the same image digests.
- ratio = amd-ubuntu / amd-cloud; **bold** = more than 5% off. For latency (TTFT, TPOT) below 1 is better.


## Analysis

- **ATOM, same images (apple-to-apple): amd-ubuntu matches amd-cloud within ±2% up to 128 users** for every config (TPOT within ±1%), so per-token GPU work is the same on both systems. Differences appear only at high load or in end-to-end throughput:
  - 256–512 users: amd-ubuntu 1.05× throughput, 0.94–0.95× TPOT.
  - `max-num-seqs 1024`: amd-ubuntu 1.15–1.26× throughput; TPOT 0.91× at 256 but 1.16–1.17× at 512–1024; not investigated.
  - Single-stream arms at 8 users: 1.14–1.15× tok/s with identical TPOT, so the gain is in prefill/scheduling, not decode.
- **AMD vLLM recipe on amd-ubuntu vs amd-cloud's best ATOM**: 1.54× at 1 user, 1.38× at 4, 1.23× at 8, 1.07× at 64, 1.04× at 128, 1.00× at 256; TPOT 0.72–0.99×. The improvement comes from speculative decoding (DSpark) at low concurrency and decode-context-parallel + CPU KV offload above; it also uses a newer stack (vLLM nightly, ROCm 10.0 user space).
- `max-num-seqs 2048` failed identically on both systems (does not fit in GPU memory).

## Results

- **apple-to-apple**: amd-cloud vs amd-ubuntu ATOM with **the same image digests** (`atom-dev:nightly_202608111555` (most experiments), `nightly_202608191459` (isl4096, repeats arm A), MAD `rocm7.2.4_..._20260727_kimi_k3` (mad, single_stream, repeats arm B, ep_matched)), same weights, flags and workload.
- **recipe**: amd-ubuntu with AMD's vLLM recipe (recipes.vllm.ai, 2026-09-25): `vllm/vllm-openai-rocm:nightly-rocm100` (digest e76a953f), DSpark speculative decoding up to C=14, decode-context-parallel 8 + CPU KV offload above, server re-tuned per concurrency, vs amd-cloud's best ATOM result at the same concurrency.

### vLLM recipe (amd-ubuntu) vs best ATOM (amd-cloud)

| C | recipe tok/s | amd-cloud ATOM best tok/s (sweep) | tok/s ratio | recipe TPOT ms | amd-cloud ATOM TPOT ms | TPOT ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71 | 46 (base) | **1.54x** | 15.36 | 21.48 | **0.72x** |
| 4 | 212 | 154 (base) | **1.38x** | 18.79 | 24.98 | **0.75x** |
| 8 | 353 | 288 (base) | **1.23x** | 22.00 | 27.02 | **0.81x** |
| 10 | 419 | — | — | 23.25 | — | — |
| 12 | 495 | — | — | 23.32 | — | — |
| 14 | 549 | — | — | 25.23 | — | — |
| 44 | 1,113 | — | — | 38.26 | — | — |
| 48 | 1,158 | — | — | 39.90 | — | — |
| 64 | 1,344 | 1,258 (base) | **1.07x** | 45.56 | 49.91 | **0.91x** |
| 70 | 1,369 | — | — | 48.84 | — | — |
| 128 | 1,865 | 1,795 (maxseqs512) | 1.04x | 66.90 | 70.71 | **0.95x** |
| 256 | 2,522 | 2,530 (maxseqs512) | 1.00x | 100.43 | 101.52 | 0.99x |

### Apple-to-apple: same images

#### base: ATOM recipe, max-num-seqs 64

**tok/s**

| max_concurrency | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 1 | 46 | 45 | 0.99x |
| 2 | 87 | 87 | 1.00x |
| 4 | 154 | 153 | 0.99x |
| 8 | 288 | 285 | 0.99x |
| 16 | 501 | 506 | 1.01x |
| 32 | 824 | 831 | 1.01x |
| 64 | 1,258 | 1,282 | 1.02x |

**TPOT ms**

| max_concurrency | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 1 | 21.5 | 21.8 | 1.01x |
| 2 | 22.6 | 22.8 | 1.01x |
| 4 | 25.0 | 25.2 | 1.01x |
| 8 | 27.0 | 27.3 | 1.01x |
| 16 | 31.2 | 30.8 | 0.99x |
| 32 | 37.8 | 37.5 | 0.99x |
| 64 | 49.9 | 49.1 | 0.98x |

#### maxseqs: max-num-seqs 256

**tok/s**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 64 | 1,237 | 1,275 | 1.03x |
| 128 | 1,792 | 1,845 | 1.03x |
| 256 | 2,482 | 2,602 | 1.05x |

**TPOT ms**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 64 | 50.0 | 49.0 | 0.98x |
| 128 | 71.0 | 68.9 | 0.97x |
| 256 | 103.7 | 98.5 | **0.95x** |

#### mad: MAD recipe, max-num-seqs 64

**tok/s**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 64 | 1,143 | 1,139 | 1.00x |
| 128 | 1,187 | 1,207 | 1.02x |
| 256 | 1,182 | 1,200 | 1.01x |

**TPOT ms**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 64 | 53.0 | 52.6 | 0.99x |
| 128 | 53.4 | 52.4 | 0.98x |
| 256 | 53.8 | 53.1 | 0.99x |

#### max-num-seqs 512

**tok/s**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 64 | 1,236 | 1,266 | 1.02x |
| 128 | 1,795 | 1,827 | 1.02x |
| 256 | 2,530 | 2,547 | 1.01x |
| 512 | 3,386 | 3,562 | **1.05x** |

**TPOT ms**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 64 | 50.1 | 48.9 | 0.98x |
| 128 | 70.7 | 68.7 | 0.97x |
| 256 | 101.5 | 99.2 | 0.98x |
| 512 | 152.8 | 143.5 | **0.94x** |

#### max-num-seqs 1024

**tok/s**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 256 | 2,028 | 2,560 | **1.26x** |
| 512 | 1,946 | 2,235 | **1.15x** |
| 1024 | 1,919 | 2,222 | **1.16x** |

**TPOT ms**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 256 | 108.4 | 99.1 | **0.91x** |
| 512 | 114.4 | 133.7 | **1.17x** |
| 1024 | 116.0 | 134.1 | **1.16x** |

#### max-num-seqs 2048

*Not runnable on 8 × MI355X at TP8, on amd-cloud (2026-08-20) and amd-ubuntu (2026-10-02, 2026-10-04) alike.* ATOM reserves Kimi-K3's KDA recurrent state (FP32) per sequence slot before the paged KV cache: 107 GiB per GPU for 2048 slots, against ~58 GiB left after the 190 GiB of weights and activations at `--gpu-memory-utilization 0.93` (ATOM: "would need 1.10"). max-num-seqs 1024 (54 GiB of state) is the largest power of two that fits.

#### isl4096: ISL 4096 / OSL 1024

**tok/s**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 64 | 1,225 | 1,206 | 0.98x |
| 128 | 1,671 | 1,675 | 1.00x |
| 256 | 2,098 | 2,115 | 1.01x |

**TTFT ms**

| conc | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---:|---:|---:|
| 64 | 319 | 320 | 1.00x |
| 128 | 472 | 469 | 0.99x |
| 256 | 532 | 556 | 1.05x |

#### single_stream: latency arms

**TPOT ms**

| arm | concurrency | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---|---:|---:|---:|
| K1_mad_default | 1 | 24.44 | 24.66 | 1.01x |
| K1_mad_default | 2 | 25.21 | 25.40 | 1.01x |
| K1_mad_default | 4 | 26.53 | 26.74 | 1.01x |
| K1_mad_default | 8 | 28.81 | 28.84 | 1.00x |
| K3_aiter_attn | 1 | 24.49 | 24.54 | 1.00x |
| K3_aiter_attn | 2 | 25.14 | 25.25 | 1.00x |
| K3_aiter_attn | 4 | 26.49 | 26.63 | 1.01x |
| K3_aiter_attn | 8 | 28.73 | 28.85 | 1.00x |
| K4_grouped_gemm | 1 | 23.78 | 23.83 | 1.00x |
| K4_grouped_gemm | 2 | 24.48 | 24.65 | 1.01x |
| K4_grouped_gemm | 4 | 25.85 | 26.02 | 1.01x |
| K4_grouped_gemm | 8 | 28.19 | 28.09 | 1.00x |

**tok/s**

| arm | concurrency | amd-cloud | amd-ubuntu | ubuntu/cloud |
|---|---|---:|---:|---:|
| K1_mad_default | 1 | 40.5 | 40.2 | 0.99x |
| K1_mad_default | 2 | 77.2 | 76.8 | 0.99x |
| K1_mad_default | 4 | 144.2 | 143.8 | 1.00x |
| K1_mad_default | 8 | 231.7 | 263.2 | **1.14x** |
| K3_aiter_attn | 1 | 40.5 | 40.4 | 1.00x |
| K3_aiter_attn | 2 | 77.4 | 77.2 | 1.00x |
| K3_aiter_attn | 4 | 144.6 | 144.1 | 1.00x |
| K3_aiter_attn | 8 | 231.8 | 263.4 | **1.14x** |
| K4_grouped_gemm | 1 | 41.7 | 41.6 | 1.00x |
| K4_grouped_gemm | 2 | 79.4 | 79.1 | 1.00x |
| K4_grouped_gemm | 4 | 148.2 | 147.7 | 1.00x |
| K4_grouped_gemm | 8 | 235.9 | 270.6 | **1.15x** |

#### repeats: mean ± std tok/s (c=256)

| config | amd-cloud | amd-ubuntu |
|---|---:|---:|
| A_original | 1,365 ± 16 (n=3) | 1,404 ± 15 (n=3) |
| B_mad | 1,180 ± 27 (n=3) | 1,212 ± 1 (n=3) |
