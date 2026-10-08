# amd-ubuntu — Kimi-K3 with AMD's recipe PDF (vLLM v0.29.0)

Recipe: `amd-kimi-k3-recipe.pdf` — `vllm/vllm-openai-rocm:v0.29.0`, TP8, one server for the sweep, `max-num-seqs 128`, `max-num-batched-tokens 4096`, `gpu-memory-utilization 0.95`, cudagraph `FULL_DECODE_ONLY`, `+fused_rms_norm_gated`, AITER MXFP4 MoE (`VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1`), no speculative decoding. Scripts: `atom/run_kimi_amdrecipe.sh` (`isl128k` on node6100, `isl1k` on node6101), weights from `/scratch/Kimi-K3`, run under apptainer.

## 1. AMD's workload (ISL/OSL 128K/1K): ours vs the PDF

*Pending: the `isl128k` arm has not produced results yet.*

## 2. Our workload (ISL/OSL 1K/1K): AMD PDF recipe vs current recipes

PDF recipe: node6101, range ratio 0.8 and 10 × C prompts like the earlier Kimi runs. Current vLLM recipe: `results/node6101/kimi-recipe/` (DSpark speculative decoding up to C=14, DCP 8 + CPU KV offload above, server re-tuned per C). ATOM: best of base / max-num-seqs 256 / 512 (`results/node6100/kimi-cloud/`). Ratios are PDF recipe / other: tok/s above 1 and TPOT below 1 favour the PDF recipe; bold = more than 5%.

| C | PDF tok/s | vLLM recipe tok/s | ATOM tok/s (config) | PDF / vLLM recipe | PDF / ATOM | PDF TPOT ms | vLLM recipe TPOT ms | ATOM TPOT ms | TPOT PDF / vLLM recipe | TPOT PDF / ATOM | PDF TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 71 | 45 (base) | **0.76x** | **1.18x** | 18.15 | 15.36 | 21.76 | **1.18x** | **0.83x** | 186 |
| 2 | 100 | — | 87 (base) | — | **1.16x** | 19.02 | — | 22.80 | — | **0.83x** | 403 |
| 4 | 183 | 212 | 153 (base) | **0.86x** | **1.20x** | 20.72 | 18.79 | 25.23 | **1.10x** | **0.82x** | 405 |
| 8 | 315 | 353 | 285 (base) | **0.89x** | **1.10x** | 23.84 | 22.00 | 27.26 | **1.08x** | **0.87x** | 408 |
| 16 | 523 | — | 506 (base) | — | 1.03x | 28.48 | — | 30.79 | — | **0.92x** | 410 |
| 32 | 852 | — | 831 (base) | — | 1.03x | 35.20 | — | 37.51 | — | **0.94x** | 412 |
| 64 | 1,228 | 1,344 | 1,282 (base) | **0.91x** | 0.96x | 49.67 | 45.56 | 49.05 | **1.09x** | 1.01x | 425 |
| 128 | 1,735 | 1,865 | 1,845 (max-num-seqs 256) | **0.93x** | **0.94x** | 71.75 | 66.90 | 68.87 | **1.07x** | 1.04x | 450 |
| 256 | 1,747 | 2,522 | 2,602 (max-num-seqs 256) | **0.69x** | **0.67x** | 71.75 | 100.43 | 98.49 | **0.71x** | **0.73x** | 73,180 |

C = 256 is above the PDF recipe's `max-num-seqs 128`, so half the requests queue; it is kept to line up with ATOM.

## Is this apple-to-apple?

- **§1 (vs the PDF): nearly.** Same image, server flags and client settings on the same GPU type. Differences: this machine, apptainer instead of docker, weights from local disk, and one short warm-up before the sweep.
- **§2 (vs current recipes): no, recipe vs recipe.** Same hardware, model, ISL/OSL, concurrency and prompt count, but each recipe has its own vLLM/ATOM version and server settings (the current vLLM recipe uses speculative decoding, the PDF recipe does not; ATOM is a different engine).

Per-arm detail: `results/<node>/kimi-amd-recipe/{isl128k,isl1k}.md`; logs: `logs/<node>/kimi-amd-recipe/atom/kimi_amdrecipe_<arm>_<ts>/`.
