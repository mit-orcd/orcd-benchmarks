# Kimi-K3 on our MI355X nodes — summary of all runs

Hand-written, 2026-10-08. One page for every Kimi-K3 serving run on node6100 / node6101 (8 × MI355X each, one node per run, TP8, weights from `/scratch/Kimi-K3`). Throughput = output tokens per second for the whole node; TPOT = median time per output token (lower is better). Ratios below 1 mean the first-named system or recipe is slower in throughput; for TPOT a ratio below 1 means it is faster.

Detail: [ubuntu/kimi.md](ubuntu/kimi.md) (ATOM and the vLLM recipe), [ubuntu/kimi-amd-recipe.md](ubuntu/kimi-amd-recipe.md) (AMD's recipe PDF), [ubuntu/kimi-recipe-old-vs-new.md](ubuntu/kimi-recipe-old-vs-new.md) (old recipes vs the new one), [vs-amd-cloud/kimi.md](vs-amd-cloud/kimi.md) (vs amd-cloud), [mi355x-vs-b200.md §6](mi355x-vs-b200.md#6-inference--kimi-k3) (vs B200). All results: [SUMMARY.md](SUMMARY.md).

## The three recipes

| Recipe | Engine and image | Main settings | Runs |
|---|---|---|---|
| **ATOM** (old) | ATOM, the same images (same digests) as amd-cloud | base `max-num-seqs` 64; variants 256 / 512 / 1024 / 2048; MAD recipe | 1K/1K, 1–512 users; ISL 4096 |
| **vLLM recipe** (old) | AMD vLLM (ROCm) recipe | DSpark speculative decoding up to 14 users; DCP 8 + CPU KV offload above; server re-tuned per load | 1K/1K, 1–256 users |
| **AMD's recipe PDF** (new) | `vllm/vllm-openai-rocm:v0.29.0` (`amd-kimi-k3-recipe.pdf`) | one server for the sweep, `max-num-seqs` 128, `max-num-batched-tokens` 4096, no speculative decoding | 128K/1K, 1–128 users (node6100); 1K/1K, 1–256 users (node6101) |

## 1. Our nodes vs amd-cloud (ATOM, same images)

**Compared: our nodes vs amd-cloud, both ATOM on the same images, same ISL/OSL 1K/1K and user counts.**

- **Our nodes are almost the same as amd-cloud up to 128 users:** throughput within ±2%, TPOT within ±1%.
- From 256 users up our nodes are faster: 1.05× at 256–512 users, 1.15–1.26× with `max-num-seqs` 1024.
- `max-num-seqs` 2048 fails on both (it does not fit in memory: ≈107 GB of cache per request against a 58 GB budget).

## 2. Our nodes vs AMD's published numbers (AMD's recipe PDF, AMD's workload)

**Compared: AMD's recipe PDF run on node6100 vs the numbers printed in the PDF, same image, server flags and client settings, ISL/OSL 128K/1K.** Metric: total tokens (input + output) per second per GPU, as in the PDF.

| Users | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ours, tok/s/GPU | 487 | 791 | 1,015 | 1,212 | 1,189 | 1,227 | 1,191 | 1,188 |
| PDF, tok/s/GPU | 874 | 1,090 | 1,235 | 1,199 | 1,179 | 1,118 | 1,096 | 1,110 |
| ours / PDF | **0.56×** | **0.73×** | **0.82×** | 1.01× | 1.01× | **1.10×** | **1.09×** | **1.07×** |

- **From 8 users up our nodes are the same as or faster than AMD's published result (1.01–1.10×).**
- **At 1–4 users our nodes are slower (0.56–0.82×).** Not explained yet. TTFT for one 128K prompt is ≈9.5 s here. Possible causes: a different warm-up (we run one short warm-up before the sweep), apptainer instead of docker, and the PDF machine's host. The GPU-bound part (8+ users) matches, so the GPUs themselves are not the cause.

## 3. AMD's recipe PDF on our workload (ISL/OSL 1K/1K)

**New recipe results on node6101** (range ratio 0.8, 10 × users prompts, like the earlier Kimi runs). The comparison with the old recipes is in its own file: [ubuntu/kimi-recipe-old-vs-new.md](ubuntu/kimi-recipe-old-vs-new.md).

| Users | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| output tok/s | 54 | 100 | 183 | 315 | 523 | 852 | 1,228 | 1,735 | 1,747 |
| TPOT median ms | 18.1 | 19.0 | 20.7 | 23.8 | 28.5 | 35.2 | 49.7 | 71.8 | 71.8 |
| TTFT median ms | 186 | 403 | 405 | 408 | 410 | 412 | 425 | 450 | 73,180 |

- Throughput grows with load up to 128 users (1,735 tok/s), with TTFT under 0.5 s.
- **At 256 users throughput stops growing and TTFT jumps to 73 s:** the recipe's `max-num-seqs` 128 makes half the requests wait. Raise it for heavier load.

## 4. MI355X vs B200 (pointer)

The full comparison is in [mi355x-vs-b200.md §6](mi355x-vs-b200.md#6-inference--kimi-k3). In short: the model fits in one MI355X node but needs two B200 nodes (16 GPUs). **With AMD's recipe PDF, MI355X serves 1.21–1.45× more tokens per GPU than B200, and one user's answer streams 1.4–1.6× faster on B200** (TPOT).

## Open questions

- AMD's recipe PDF at 1–4 users on the 128K/1K workload is 18–44% below the PDF's numbers (§2).
- Why single-user TPOT on MI355X is higher than on B200 at the same memory bandwidth on paper (not profiled).
