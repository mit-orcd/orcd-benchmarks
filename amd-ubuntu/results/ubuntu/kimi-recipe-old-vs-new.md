# amd-ubuntu — Kimi-K3: old recipes vs the new AMD recipe 2026-10

**Compared: the new AMD recipe 2026-10 (`amd-kimi-k3-recipe.pdf`, vLLM v0.29.0, no speculative decoding) vs the two old recipes (the vLLM recipe with speculative decoding, and ATOM), all on our nodes, ISL/OSL 1K/1K, 8 GPUs (TP8).** Recipe vs recipe, not hardware. New-recipe results alone: [kimi-amd-recipe.md](kimi-amd-recipe.md).

new AMD recipe 2026-10: node6101, range ratio 0.8 and 10 × C prompts like the earlier Kimi runs. Old vLLM recipe: `results/node6101/kimi-recipe/` (DSpark speculative decoding up to C=14, DCP 8 + CPU KV offload above, server re-tuned per C). ATOM: best of base / max-num-seqs 256 / 512 (`results/node6100/kimi-cloud/`). Ratios are new AMD recipe 2026-10 / old recipe: tok/s above 1 and TPOT below 1 favour the new AMD recipe 2026-10; bold = more than 5%.

| C | new tok/s | old vLLM recipe tok/s | ATOM tok/s (config) | new / old vLLM recipe | new / ATOM | new TPOT ms | old vLLM recipe TPOT ms | ATOM TPOT ms | TPOT new / old vLLM recipe | TPOT new / ATOM | new TTFT ms |
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

C = 256 is above the new AMD recipe 2026-10's `max-num-seqs 128`, so half the requests queue; it is kept to line up with ATOM.

## Is this apple-to-apple?

**No, recipe vs recipe.** Same hardware, model, ISL/OSL, concurrency and prompt count, but each recipe has its own vLLM/ATOM version and server settings (the old vLLM recipe uses speculative decoding, the new AMD recipe 2026-10 does not; ATOM is a different engine).

## Reading

- **Old vLLM recipe vs new AMD recipe 2026-10: the old vLLM recipe is faster at every load where both ran** (new / old 0.76x at 1 user, 0.86–0.93x at 4–128, 0.69x at 256); its speculative decoding gives the largest gain at low load.
- **ATOM vs new AMD recipe 2026-10: the new AMD recipe 2026-10 is faster up to 32 users** (1.10–1.20x at 1–8 users, 1.03x at 16–32) and slightly slower at 64–128 (0.94–0.96x).
- **At 256 users the new AMD recipe 2026-10 falls behind both (0.67–0.69x)**: its `max-num-seqs 128` makes half the requests wait.
- **Fastest per load on our nodes:** old vLLM recipe at 1, 4, 8, 64 and 128 users; new AMD recipe 2026-10 at 2, 16 and 32 (the old vLLM recipe was not run there); ATOM (`max-num-seqs` 256–512) at 256 users and above.
- **Suggested setup:** the old vLLM recipe for interactive use (1–128 users); ATOM with `max-num-seqs` 256–512 for 256 users or more. The new AMD recipe 2026-10 is a good single setting for 2–32 users without speculative decoding; raise its `max-num-seqs` above 128 for heavier load.
