# amd-ubuntu — Kimi-K3: ATOM vs vLLM recipe 1 vs vLLM recipe 2

**Compared: the three Kimi-K3 serving recipes run on our nodes, all with ISL/OSL 1K/1K on one node of 8 × MI355X (TP8).** Recipe vs recipe, not hardware.

## The three recipes

- **ATOM:** AMD's ATOM inference engine (not vLLM), run with the same container images (same digests) as amd-cloud and the MAD benchmark recipe. Base setting `max-num-seqs` 64; variants with 256 and 512 for heavy load. No speculative decoding. Detail: [kimi.md](kimi.md).
- **vLLM recipe 1:** AMD's earlier vLLM (ROCm) recipe for Kimi-K3 from recipes.vllm.ai (`atom/run_kimi_recipe.sh`). Uses DSpark speculative decoding up to 14 users, and DCP 8 with CPU KV-cache offload above; the server is restarted with settings tuned for each load. Detail: [kimi.md](kimi.md).
- **vLLM recipe 2:** AMD's newer Kimi-K3 recipe from `amd-kimi-k3-recipe.pdf` (2026-10): the stock `vllm/vllm-openai-rocm:v0.29.0` image (its own ROCm 7.2.3) with AMD's AITER kernels, one server for the whole sweep, `max-num-seqs` 128, `max-num-batched-tokens` 4096, no speculative decoding. Detail: [kimi-amd-recipe.md](kimi-amd-recipe.md).

vLLM recipe 2: node6101, range ratio 0.8 and 10 × C prompts like the earlier Kimi runs. vLLM recipe 1: `results/node6101/kimi-recipe/` (DSpark speculative decoding up to C=14, DCP 8 + CPU KV offload above, server re-tuned per C). ATOM: best of base / max-num-seqs 256 / 512 (`results/node6100/kimi-cloud/`). Ratios are vLLM recipe 2 / the other recipe: tok/s above 1 and TPOT (time per output token) below 1 favour vLLM recipe 2; bold = more than 5%.

### Throughput (output tokens per second, higher is better)

| Users (C) | vLLM recipe 2 | vLLM recipe 1 | ATOM (config) | recipe 2 / recipe 1 | recipe 2 / ATOM |
|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 71 | 45 (base) | **0.76x** | **1.18x** |
| 2 | 100 | — | 87 (base) | — | **1.16x** |
| 4 | 183 | 212 | 153 (base) | **0.86x** | **1.20x** |
| 8 | 315 | 353 | 285 (base) | **0.89x** | **1.10x** |
| 16 | 523 | — | 506 (base) | — | 1.03x |
| 32 | 852 | — | 831 (base) | — | 1.03x |
| 64 | 1,228 | 1,344 | 1,282 (base) | **0.91x** | 0.96x |
| 128 | 1,735 | 1,865 | 1,845 (max-num-seqs 256) | **0.93x** | **0.94x** |
| 256 | 1,747 | 2,522 | 2,602 (max-num-seqs 256) | **0.69x** | **0.67x** |

### Latency (median, ms, lower is better)

| Users (C) | TPOT (vLLM recipe 2) | TPOT (vLLM recipe 1) | TPOT (ATOM) | TPOT recipe 2 / recipe 1 | TPOT recipe 2 / ATOM | TTFT (vLLM recipe 2) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.15 | 15.36 | 21.76 | **1.18x** | **0.83x** | 186 |
| 2 | 19.02 | — | 22.80 | — | **0.83x** | 403 |
| 4 | 20.72 | 18.79 | 25.23 | **1.10x** | **0.82x** | 405 |
| 8 | 23.84 | 22.00 | 27.26 | **1.08x** | **0.87x** | 408 |
| 16 | 28.48 | — | 30.79 | — | **0.92x** | 410 |
| 32 | 35.20 | — | 37.51 | — | **0.94x** | 412 |
| 64 | 49.67 | 45.56 | 49.05 | **1.09x** | 1.01x | 425 |
| 128 | 71.75 | 66.90 | 68.87 | **1.07x** | 1.04x | 450 |
| 256 | 71.75 | 100.43 | 98.49 | **0.71x** | **0.73x** | 73,180 |

C = 256 is above vLLM recipe 2's `max-num-seqs 128`, so half the requests queue; it is kept to line up with ATOM.

## AMD's workload (ISL/OSL 128K/1K): AMD's published numbers vs our three recipes

**Compared: the same workload and metric for all columns** — AMD's 128K/1K workload from `amd-kimi-k3-recipe.pdf` (input 128,000 and output 1,000 tokens, range ratio 0.2, 10 × users prompts), metric = total tokens (input + output) per second per GPU, as in the PDF. AMD published only vLLM recipe 2. Our three recipes ran on one node of 8 × MI355X (vLLM recipe 2: node6100, 2026-10-07; vLLM recipe 1 and ATOM: 2026-10-08/09, `atom/run_kimi_128k.sh`). Ratios = ours / AMD's published number (above 1 = ours faster).

| Users | AMD published (vLLM recipe 2) | ours (vLLM recipe 2) | ours (vLLM recipe 1) | ours (ATOM) | vLLM recipe 2 / AMD | vLLM recipe 1 / AMD | ATOM / AMD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 874 | 487 | 573 | 483 | **0.56x** | **0.66x** | **0.55x** |
| 2 | 1,090 | 791 | 840 | 1,123 | **0.73x** | **0.77x** | 1.03x |
| 4 | 1,235 | 1,015 | 1,076 | 1,360 | **0.82x** | **0.87x** | **1.10x** |
| 8 | 1,199 | 1,212 | 1,299 | 1,630 | 1.01x | **1.08x** | **1.36x** |
| 16 | 1,179 | 1,189 | 1,619 | 1,775 | 1.01x | **1.37x** | **1.51x** |
| 32 | 1,118 | 1,227 | 1,730 | 1,926 | **1.10x** | **1.55x** | **1.72x** |
| 64 | 1,096 | 1,191 | 1,819 | 1,947 | **1.09x** | **1.66x** | **1.78x** |
| 128 | 1,110 | 1,188 | 1,823 | 1,966 | **1.07x** | **1.64x** | **1.77x** |

## Is this apple-to-apple?

**No, recipe vs recipe.** Same hardware, model, ISL/OSL, concurrency and prompt count, but each recipe has its own vLLM/ATOM version and server settings (vLLM recipe 1 uses speculative decoding, vLLM recipe 2 does not; ATOM is a different engine).

## Reading

- **vLLM recipe 1 vs vLLM recipe 2: vLLM recipe 1 is faster at every load where both ran** (recipe 2 / recipe 1: 0.76x at 1 user, 0.86–0.93x at 4–128, 0.69x at 256); its speculative decoding gives the largest gain at low load.
- **ATOM vs vLLM recipe 2: vLLM recipe 2 is faster up to 32 users** (1.10–1.20x at 1–8 users, 1.03x at 16–32) and slightly slower at 64–128 (0.94–0.96x).
- **At 256 users vLLM recipe 2 falls behind both (0.67–0.69x)**: its `max-num-seqs 128` makes half the requests wait.
- **Fastest per load on our nodes:** vLLM recipe 1 at 1, 4, 8, 64 and 128 users; vLLM recipe 2 at 2, 16 and 32 (vLLM recipe 1 was not run there); ATOM (`max-num-seqs` 256–512) at 256 users and above.
- **Suggested setup:** vLLM recipe 1 for interactive use (1–128 users); ATOM with `max-num-seqs` 256–512 for 256 users or more. vLLM recipe 2 is a good single setting for 2–32 users without speculative decoding; raise its `max-num-seqs` above 128 for heavier load.
