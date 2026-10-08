# amd-ubuntu — Kimi-K3 with vLLM recipe 2 (vLLM v0.29.0)

Recipe: `amd-kimi-k3-recipe.pdf` — `vllm/vllm-openai-rocm:v0.29.0`, TP8, one server for the sweep, `max-num-seqs 128`, `max-num-batched-tokens 4096`, `gpu-memory-utilization 0.95`, cudagraph `FULL_DECODE_ONLY`, `+fused_rms_norm_gated`, AITER MXFP4 MoE (`VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1`), no speculative decoding. Scripts: `atom/run_kimi_amdrecipe.sh` (`isl128k` on node6100, `isl1k` on node6101), weights from `/scratch/Kimi-K3`, run under apptainer.

## 1. AMD's workload (ISL/OSL 128K/1K): our nodes vs AMD's published numbers

Our runs already use this same recipe: vLLM recipe 2 from `amd-kimi-k3-recipe.pdf` (same image `vllm/vllm-openai-rocm:v0.29.0`, server flags and client settings). AMD's published numbers are the results table printed in that PDF.

Total tok/s per GPU = total_token_throughput / 8 (the PDF's metric). Ours: node6100. Apple-to-apple: same image, server flags and client settings; different machine.

| C | ours total tok/s/GPU | PDF total tok/s/GPU | ours / PDF | out tok/s | TTFT med ms | TPOT med ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 487 | 874 | **0.56x** | 34.4 | 9,522 | 19.26 |
| 2 | 791 | 1,090 | **0.73x** | 47.3 | 11,189 | 30.36 |
| 4 | 1,015 | 1,235 | **0.82x** | 64.3 | 11,865 | 49.59 |
| 8 | 1,212 | 1,199 | 1.01x | 75.4 | 12,467 | 90.11 |
| 16 | 1,189 | 1,179 | 1.01x | 73.6 | 45,426 | 171.72 |
| 32 | 1,227 | 1,118 | **1.10x** | 76.3 | 254,051 | 162.69 |
| 64 | 1,191 | 1,096 | **1.09x** | 73.3 | 694,693 | 173.36 |
| 128 | 1,188 | 1,110 | **1.07x** | 73.6 | 1,572,110 | 172.52 |

**Why slower at 1–4 users (not proven yet):**
- **With 1 user, most of the time is spent generating the output one token at a time.** Per request (121K in, 1,082 out): 10.6 s to process the prompt + 20.8 s to generate (19.3 ms per token) = 31.4 s.
- **AMD's 874 tok/s/GPU means ≈17.5 s per request.** Our token generation alone (20.8 s) takes longer than that, so AMD's run must have had about half our time per output token (≈10 ms).
- **The likely cause is a fixed overhead per generation step, not GPU speed.** Our time per output token barely depends on context length (18.1 ms at 1K, 19.3 ms at 128K), and with 8+ users, where GPU compute dominates, we match AMD (1.01–1.10×). The same overhead is why one B200 user streams faster.
- **Ruled out:** recipe settings (same image, environment, server and client flags as the PDF), host CPU settings (performance governor, CPU sleep states off, IOMMU passthrough, NUMA balancing off) and ROCm version (the image brings its own ROCm 7.2.3).
- **Not yet tested:** apptainer instead of docker, AMD's machine (driver, firmware, host CPU), and AMD's warm-up. Next steps: profile one generation step (rocprofv3) to see if the GPUs wait between steps, and ask AMD for their TTFT and TPOT at 1 user.

## 2. Our workload (ISL/OSL 1K/1K)

node6101, range ratio 0.8 and 10 × C prompts like the earlier Kimi runs. Comparison with ATOM and vLLM recipe 1: [kimi-recipe-old-vs-new.md](kimi-recipe-old-vs-new.md).

| C | out tok/s | total tok/s | TTFT med ms | TTFT p99 ms | TPOT med ms | TPOT p99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 86 | 186 | 2,210 | 18.15 | 18.36 |
| 2 | 100 | 217 | 403 | 548 | 19.02 | 19.41 |
| 4 | 183 | 352 | 405 | 1,283 | 20.72 | 21.33 |
| 8 | 315 | 627 | 408 | 770 | 23.84 | 25.67 |
| 16 | 523 | 1,051 | 410 | 1,479 | 28.48 | 30.59 |
| 32 | 852 | 1,695 | 412 | 2,547 | 35.20 | 39.32 |
| 64 | 1,228 | 2,494 | 425 | 4,566 | 49.67 | 53.81 |
| 128 | 1,735 | 3,474 | 450 | 9,166 | 71.75 | 76.66 |
| 256 | 1,747 | 3,548 | 73,180 | 97,734 | 71.75 | 77.76 |

C = 256 is above the recipe's `max-num-seqs 128`, so half the requests queue (TTFT jumps).

## Is this apple-to-apple?

- **§1 (vs the PDF): nearly.** Same image, server flags and client settings on the same GPU type. Differences: this machine, apptainer instead of docker, weights from local disk, and one short warm-up before the sweep.

Per-arm detail: `results/<node>/kimi-amd-recipe/{isl128k,isl1k}.md`; logs: `logs/<node>/kimi-amd-recipe/atom/kimi_amdrecipe_<arm>_<ts>/`.
