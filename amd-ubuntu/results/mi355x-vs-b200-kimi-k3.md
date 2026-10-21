# Kimi-K3: MI355X vs B200, cross-check against SemiAnalysis InferenceX

SemiAnalysis publish continuous Kimi-K3 benchmarks at <https://inferencex.semianalysis.com> (InferenceX), and open-source the harness (Apache-2.0, `github.com/SemiAnalysisAI/InferenceX`). The InferenceX configs are in `../../b200-kimi/semianalysis-ref/`. InferenceX's **B200 recipe uses TP8 × PP2 on 2 nodes — the same layout as our B200 run**, which makes the methodology directly checkable.

**Ours** = the Kimi-K3 results in [mi355x-vs-b200.md](mi355x-vs-b200.md) §5:
- **16 × B200** (two nodes, TP8 × PP2), vLLM `vllm/vllm-openai:kimi-k3`, no speculative decoding.
- **8 × MI355X, vLLM recipe 1** (one node, TP8): AMD's vLLM (ROCm) recipe from [recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x](https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x) (2026-09-25), `vllm/vllm-openai-rocm:nightly-rocm100`. DSpark speculative decoding (MTP, §6) up to 14 users, DCP 8 + CPU KV-cache offload above, server re-tuned for each load. Detail: `node6101/kimi-recipe/kimi-k3-recipe.md`.
- **8 × MI355X, ATOM + MTP** (one node, TP8): our ATOM flags (max-num-seqs 64), image `rocm/atom-dev:nightly_202610011450` plus DSpark MTP (`--method dspark`, `Inferact/Kimi-K3-DSpark`), draft length 6 at 1 user, 2 above, as in the InferenceX ATOM recipe (our ATOM baseline image, `nightly_202608111555`, crashes with this draft). Detail: `ubuntu/kimi-atom-mtp.md`.
- Both: ISL/OSL 1024/1024, `--ignore-eos`, random synthetic prompts, 10 × users prompts, `vllm bench serve`.

TPOT = time per output token (how fast one user's answer streams). TTFT = time to first token (how long until the answer starts). Conc = concurrency (number of users).

## Contents

1. [What InferenceX independently confirms](#1-what-inferencex-independently-confirms)
2. [What differs — why the numbers are NOT directly comparable](#2-what-differs--why-the-numbers-are-not-directly-comparable)
3. [Aggregate throughput — InferenceX vs ours](#3-aggregate-throughput--inferencex-vs-ours)
4. [Per-user tok/s — InferenceX vs ours](#4-per-user-toks--inferencex-vs-ours)
5. [One agent turn, end to end](#5-one-agent-turn-end-to-end)
6. [MTP — what it is, and how to enable it](#6-mtp--what-it-is-and-how-to-enable-it)

## 1. What InferenceX independently confirms

| Fact | **Applies to** | Ours (measured) | InferenceX (stated in config) |
|---|---|---|---|
| Checkpoint size | **both** | 1,560,936,091,448 B = 1.561 TB / 1.420 TiB, 96 shards | *"1.561 TB decimal (1.420 TiB, 96 safetensors)"* |
| Does not fit one 8×B200 node | **B200** | 1561 GB vs 1538 GB — 23 GB short | *"does not fit one 8xB200 node, so TP8 shards … and PP2 splits the 93 layers"* |
| Layout forced to TP8 × PP2, 16 GPUs | **B200** | yes | `tensor-parallel-size: 8`, `pipeline-parallel-size: 2`, `agg_nodes: 2` |
| `gpu-memory-utilization` 0.90 not 0.95 | **B200** | 0.90 | 0.90, and the *same reason*: *"the flashinfer trtllm MXFP4 MoE kernel allocates a ~1.6 GiB runtime workspace OUTSIDE vLLM's memory pool … at 0.95 a 178 GiB B200 … OOMs"* |
| Usable HBM per GPU | **B200** | 178.35 GiB (nvidia-smi) | *"a 178 GiB B200"* |
| Fits on ONE node at TP8 | **MI355X** | yes, 288 GB/GPU | *"~195 GB/GPU across 8 GPUs of the 288 GB part; TP=4 … cannot load"* |
| Expert parallelism off | **both** | EP off | *"Plain TP (NOT TEP): expert parallelism is deliberately off"* |
| Image, load format, autotune, batched-token cap | **B200** | `vllm/vllm-openai:kimi-k3`, `fastsafetensors`, autotune off, 8192 | identical on all four |
| Speculative decoding on MI355X | **MI355X** | DSpark (`method: dspark`), from the upstream vLLM recipe | DSpark MTP (`SPEC_DECODE true`), *"the upstream vLLM recipe's amd block"* |
| Native context, KV-cache type | **MI355X** | `max-model-len` 1M (1,048,576), fp8 KV cache | `MAX_MODEL_LEN 1M`, `KV_CACHE_DTYPE fp8` |

Two independent teams, different clusters, same conclusions — including the exact checkpoint byte count and the non-obvious 0.90 memory-utilisation workaround.

## 2. What differs — why the numbers are NOT directly comparable

| Dimension | **Applies to** | Ours | InferenceX | Effect |
|---|---|---|---|---|
| **Workload** | **both** | fixed ISL/OSL 1024/1024, `--ignore-eos`, random synthetic | **AgentX agentic trace replay**, real multi-turn traces, 1M+ context | **largest difference.** InferenceX traces have long shared prefixes and huge context; ours is a fixed, cache-hostile synthetic shape |
| **Prefix caching** | **both** | B200 **off**; MI355X on (recipe setting), random prompts | **on** (default, kept for trajectory reuse) | InferenceX reuses KV across turns; our random prompts give it little to reuse. Big throughput swing on agentic traffic |
| `max-model-len` | **B200** | 16384 | native **1M** (unset) | InferenceX pays a far larger KV footprint per sequence |
| `max-num-seqs` | **B200** | **64** (fixed) | let vLLM choose | ours deliberately caps the batch; `../../b200-kimi/results/kimi-k3-base-b200.md` §3.2 shows that cap is what binds our throughput |
| Benchmark client | **both** | `vllm bench serve` | `aiperf` + trace replay | different measurement harness |
| **Spec decoding (MTP)** | **B200** | **off** | both: no MTP and **DSpark MTP on** | InferenceX's B200 +MTP arm gets a lever ours does not use — see §6 |
| **Spec decoding (MTP)** | **MI355X** | **DSpark MTP on up to 14 users** (draft length 7, 5, 4, 3 as load rises), off from 44 users | **DSpark MTP on** (`SPEC_NUM_TOKENS 2`) | both MI355X arms use MTP at low load; ours uses a longer draft |
| `max-num-seqs` | **MI355X** | 2 × users (2–512, recipe setting) | 128 | ours re-tunes the batch for each load |
| `gpu-memory-utilization` | **MI355X** | 0.90 | 0.95 | small KV-cache difference |
| Serving engine | **MI355X** | vLLM ROCm (vLLM recipe 1) | vLLM ROCm *and* an ATOM variant | our MI355X column lines up with InferenceX's MI355X vLLM column |
| Acceptance length (ATOM +MTP) | **MI355X** | real (measured draft/target agreement) | **forced**: `--spec-decode-acceptance-length` 3.75 at 1 user, 2.51 above (*"the rejection sampler ignores the real draft/target agreement and force-accepts draft tokens"*) | the InferenceX ATOM +MTP numbers do not come from real acceptance |

**Precision is the same, despite the labels.** InferenceX's config says `precision: "fp4"` and ours says MXFP4 — the same thing. Both serve the native `moonshotai/Kimi-K3` MXFP4 checkpoint (`mxfp4-pack-quantized`, 4-bit routed experts with e8m0 scales); neither re-quantises. "FP4" on the InferenceX dashboard is the checkpoint's own format, not a separate NVFP4 conversion.

**Why ours looks much better than InferenceX — the workload, not the hardware.** Same model, same B200 image and layout, but a different job:

- **Ours:** random prompts, 1024 tokens in, 1024 out — short and the same every time.
- **InferenceX:** real agent traces — long, growing context (up to 1M tokens), mostly input, short outputs. Each output token reads far more KV cache, and prefill from other users takes GPU time from decode.
- **At 1 user the two agree:** B200 89.0 (ours) vs 81.9 (InferenceX) tok/s per user. This is the fair check, and it shows our setup is sound.
- **With more users the gap grows:** 53.6 vs 8.6 at 16 users, 39.6 vs 3.8 at 32. Long contexts crowd out decode; our short requests never hit this.
- **Aggregate throughput counts output tokens only**, so InferenceX's input-heavy requests look slow even when the GPUs are busy.
- **InferenceX's B200 columns mix two runs:** from 48 users the no-MTP rows use a newer nightly image (`nightly-dev-x86_64-cu13.0.1-728d3ad`), hence the jump from 60 tok/s at 32 users to 838 at 48.

So "ours is faster" at high load is not a valid conclusion — ours is an easier workload. A like-for-like comparison would need the InferenceX agent traces replayed on our nodes.

## 3. Aggregate throughput — InferenceX vs ours

The InferenceX API carries aggregate output throughput (`output_tput_tps`) at every concurrency point — the same quantity as our "Throughput (tok/s)" column ([mi355x-vs-b200.md](mi355x-vs-b200.md) §5.1). InferenceX's **B200 arms are 16 GPUs, TP8 × PP2 across 2 nodes — the same layout as ours**; its MI355X arms are 8 GPUs on one node, as is ours.

| Conc | **Ours** B200 TP8×PP2 (no MTP) | **InferenceX** B200 TP8×PP2 (no MTP) | **InferenceX** B200 +MTP | **Ours** MI355X vLLM recipe 1 (+MTP up to 14 users) | **InferenceX** MI355X vLLM +MTP | **Ours** MI355X ATOM (no MTP) | **Ours** MI355X ATOM +MTP | **InferenceX** MI355X ATOM +MTP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 86.7 | 59.9 | 118.1 | 70.7 | 67.4 | 63.1 | 197.4 | 91.0 |
| 2 | 165.2 | 72.1 | 144.5 | — | — | — | 280.7 | — |
| 4 | 280.8 | 76.0 | 190.3 | 212.5 | 115.7 | — | 464.1 | 137.1 |
| 8 | 475.3 | 99.7 | 338.3 | 353.0 | 169.5 | 430.8 | 811.5 | 233.0 |
| 10 | — | — | — | 418.6 | 235.0 | — | 695.5 | 293.1 |
| 16 | 801.8 | 92.4 | 518.0 | — | — | — | 1,117.0 | — |
| 32 | 1,209.3 | 60.2 | 768.0 | — | — | 1,084.7 | 1,810.2 | — |
| 48 | — | 838.2 | — | 1,157.9 (no MTP) | — | — | 2,149.0 | — |
| 64 | 1,696.4 | — | — | 1,344.2 (no MTP) | — | — | 2,463.1 | — |
| 72 | — | 1,021.6 | — | — | — | — | — | — |
| 96 | — | 1,159.4 | — | — | — | — | — | — |

*Ours MI355X ATOM (no MTP) = the no-MTP arm of the same ATOM + MTP run (same image, `nightly_202610011450`), measured at 1, 8 and 32 users only.*

### Why ours leads here but trails on per-user tok/s (§4)

The two metrics have different denominators:

- **Per-user tok/s** = `1 / TPOT` — the rate a request streams *while it is decoding*. Measured directly (InferenceX's `median_intvty`, our median TPOT).
- **Aggregate tok/s** = output tokens ÷ total wall clock — which includes every second spent on **prefill**, producing no output at all.

So `aggregate ÷ concurrency` is always ≤ per-user tok/s, and the gap is the **decode duty cycle** — the fraction of the run that was decode rather than prefill:

| | Conc | Aggregate | ÷ conc | Per-user (measured) | Duty cycle |
|---|---:|---:|---:|---:|---:|
| **Ours** B200, 1024/1024 | 64 | 1,696.4 | 26.5 | 27.9 | **0.95** |
| **Ours** MI355X vLLM recipe 1, 1024/1024 | 64 | 1,344.2 | 21.0 | 21.9 | **0.96** |
| **Ours** MI355X ATOM +MTP, 1024/1024 | 64 | 2,463.1 | 38.5 | 41.7 | **0.92** |
| **InferenceX** B200 +MTP, agentic | 32 | 768.0 | 24.0 | 46.5 | **0.52** |

InferenceX traces average **~138k input tokens** per request against a handful of output tokens, so half the InferenceX wall clock is prefill; our `--ignore-eos` 1024/1024 shape keeps nearly every slot decoding continuously. Neither number is a hardware verdict:

- **Our aggregate lead is a workload artifact.** Counting *all* tokens processed, our c=64 runs move ~3,389.7 tok/s on B200 and 2,729.8 tok/s on MI355X (input + output); the InferenceX B200 c=32 run moves **111,970**. InferenceX does far more token work — it just is not output. (InferenceX's `tput_per_gpu` is computed on that total, which is why it looks so large.)
- **InferenceX's per-user lead is a real software lever** — MTP, which our B200 run does not use. Turning it on would raise *both* of our B200 columns (§6).

**Which view is "real world" depends on the workload, not the user count.** One interactive chat issues one request at a time and feels per-user tok/s; a batch job or an agentic app fanning out parallel calls fills the server itself and gets the aggregate rate. The InferenceX data makes the same point from the other side: its aggregate column is a *capacity* number measured under agentic load, which is why it is dominated by prefill and says almost nothing about how fast a single agent turn streams. For that, read §4.

**The InferenceX B200 no-MTP column is not one curve.** It falls to 60.2 at c=32 and then jumps to 838.2 at c=48 because the c=1–32 and c=48–96 points come from different recipe families (the low-concurrency records carry no `recipe_fingerprint`; the high-concurrency ones do). Do not read a scaling trend down that column.

## 4. Per-user tok/s — InferenceX vs ours

Retrieved from the InferenceX public API (`/api/v1/benchmarks?model=Kimi-K3`); an extracted CSV is in `../../b200-kimi/semianalysis-ref/`. The InferenceX **"Interactivity"** metric is exactly our per-user tok/s — verified against its own fields: `mean_tpot` 0.00454 s → 1/0.00454 = 220.3 = `mean_intvty`. Same definition, `1 / TPOT`.

| Conc | **Ours** B200 TP8×PP2 (no MTP) | **InferenceX** B200 TP8×PP2 (no MTP) | **InferenceX** B200 +MTP | **Ours** MI355X vLLM recipe 1 (+MTP up to 14 users) | **InferenceX** MI355X vLLM +MTP | **Ours** MI355X ATOM (no MTP) | **Ours** MI355X ATOM +MTP | **InferenceX** MI355X ATOM +MTP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **89.0** | 81.9 | **221.7** | 65.1 | 84.0 | 64.0 | 256.8 | 127.2 |
| 2 | 84.4 | 74.8 | 219.3 | — | — | — | 161.5 | — |
| 4 | 73.4 | 54.1 | 203.3 | 53.2 | 62.5 | — | 138.5 | 88.7 |
| 8 | 64.9 | 21.1 | 154.1 | 45.5 | 41.4 | 56.4 | 112.2 | 64.7 |
| 10 | — | — | — | 43.0 | 35.7 | — | 78.4 | 62.7 |
| 16 | 53.6 | 8.6 | 77.7 | — | — | — | 78.2 | — |
| 32 | 39.6 | 3.8 | 46.5 | — | — | 35.3 | 61.7 | — |
| 64 | 27.9 | — | — | 21.9 (no MTP) | — | — | 41.7 | — |

*Ours MI355X ATOM (no MTP) = the no-MTP arm of the same ATOM + MTP run (same image, `nightly_202610011450`), measured at 1, 8 and 32 users only.*

**"no MTP" = plain decoding, one token per forward pass. "+MTP" = speculative decoding on: a draft model proposes several tokens and the big model verifies them in one pass.**

> ⚠️ **Every one of the InferenceX MI355X runs uses MTP — InferenceX publishes no MI355X result without it — so the InferenceX MI355X columns can never be compared against the B200 no-MTP columns on equal footing.** (Verified in the InferenceX API data: all 8 MI355X records are `spec_method: mtp`; B200 has both, 9 `none` and 6 `mtp`.) The same holds for our MI355X column up to 14 users.

Units: output tokens/s delivered to a single request. InferenceX's are `median_intvty`; ours are `1000 / median TPOT` (with speculative decoding, TPOT counts accepted tokens, so it already includes the draft's benefit). **Read across rows with care — see §2.** The columns differ in workload (agentic traces vs fixed 1024/1024), prefix caching and context (1M vs 16K on B200), so this is not a like-for-like ranking.

**What the table shows, in four points:**

1. **Our B200 no-MTP number matches InferenceX's at c=1** (89.0 vs 81.9) — an independent check that our TP8×PP2 setup is performing normally, not misconfigured.
2. **MTP is worth ~2.7× at c=1 on B200**, in the InferenceX data, same hardware — the single largest lever in this table. §6 shows how to get it.
3. **Our MI355X vLLM recipe 1 (+MTP) lands close to InferenceX's MI355X vLLM +MTP**: 0.77× at c=1 (65.1 vs 84.0), 0.85× at c=4, and faster at c=8–10 (1.10–1.20×). With MTP on MI355X and off on B200, our B200 still streams 1.37× faster at c=1 (89.0 vs 65.1). The biggest differentiator in the whole table is MTP — a **software** feature both platforms can run, not a silicon difference.
4. **MTP is worth 4.0× at c=1 on our MI355X ATOM** (256.8 vs 64.0 without MTP, real acceptance). InferenceX's MI355X ATOM +MTP gets 127.2 at c=1, but with a forced acceptance length (§2).

## 5. One agent turn, end to end

A user waits for a whole turn, not for a token. Modelled here as one realistic agent request — **plan (200 tok) → act: launch 8 independent tool calls (500 tok each, hidden) → stream the answer (800 tok, shown)** — priced with each system's own measured TTFT and per-user rate. **English runs ~1.3 tokens per word** (≈0.75 words per token), so an 800-token answer is ~600 words and the print rates below are `tok/s × 0.75`.

The three steps play different roles: the model *thinks* (plan), then *acts* (tool calls), then *writes* (answer). Only *act* runs in parallel — it is the only step where the work naturally splits into independent pieces; you cannot parallelise "write one coherent plan" or "write one coherent answer" the same way.

**Act** = the agent **launches 8 independent tasks at once** — searches, file reads, subagents, tool calls — instead of running them one after another. To the server that is 8 concurrent requests from a single agent, which is why the concurrency axis in §3 and §4 maps onto agent behaviour at all. It only works when no task needs another's result. Real agents launch ~2–8 at a time; **8 is used here because it is a measured point on every sweep in §3 and §4**.

**Which number governs each step:**

| Step | Conc | Rate that governs it | Ours 16× B200 | Ours 8× MI355X (vLLM recipe 1) | Read it from |
|---|---:|---|---:|---:|---|
| 1. Plan / pick tools — *think* | **1** | per-user tok/s | 89.0 tok/s | 65.1 tok/s | §4, c=1 row |
| 2. Eight parallel tool calls — *act* | **8** | per-user tok/s at c=8 — eight streams run at once, so the *session* rate is 8× that | 64.9 each → **519.3 tok/s** session | 45.5 each → **363.6 tok/s** session | §4, c=8 row; the session rate is §3's aggregate column |
| 3. Stream the answer — *write* | **1** | per-user tok/s | 89.0 tok/s | 65.1 tok/s | §4, c=1 row |

Each step costs `TTFT + (tokens per stream − 1) ÷ per-user tok/s`. Only the **per-user** rate enters the arithmetic — the parallel act step is already accounted for by the fact that eight streams run concurrently, so §3's aggregate is the same information viewed from the server side, not a second speedup to multiply in. (*Close to §3's aggregate, not exactly equal*: `8 × per-user` is a steady-state rate, while the aggregate column is measured tokens ÷ wall clock, so it carries the prefill time too.)

| System | Plan | 8 tool calls | Answer | **Whole turn** | First word at | Print rate |
|---|---:|---:|---:|---:|---:|---:|
| **Ours** 16× B200 (no MTP) | 2.5 s | 7.9 s | 9.2 s | **19.6 s** | 10.6 s | 67 words/s |
| **Ours** 8× MI355X vLLM recipe 1 (+MTP) | 3.3 s | 11.3 s | 12.5 s | **27.0 s** | 14.7 s | 49 words/s |
| **Ours** 8× MI355X ATOM (no MTP) | 3.3 s | 9.0 s | 12.7 s | **25.0 s** | 12.5 s | 48 words/s |
| **Ours** 8× MI355X ATOM +MTP | 0.9 s | 4.6 s | 3.3 s | **8.9 s** | 5.8 s | 193 words/s |
| **InferenceX** 16× B200 (no MTP) | 6.9 s | 30.3 s | 14.3 s | **51.6 s** | 41.8 s | 61 words/s |
| **InferenceX** 16× B200 +MTP | 2.0 s | 4.1 s | 4.8 s | **10.9 s** | 7.3 s | 166 words/s |
| **InferenceX** 8× MI355X ATOM +MTP | 2.4 s | 8.5 s | 7.1 s | **18.0 s** | 11.7 s | 95 words/s |
| **InferenceX** 8× MI355X vLLM +MTP | 3.8 s | 13.7 s | 10.9 s | **28.4 s** | 18.9 s | 63 words/s |

*Worked example, the first row.* §4 gives 89.0 tok/s at c=1 and 64.9 at c=8. Step 1 = 0.23 + 199/89.0 = 2.5 s. Step 2 = 0.23 + 499/64.9 = 7.9 s for all eight calls together — a session rate of 504.9 tok/s (the steady-state 519.3, diluted by the 0.23 s prefill), which is §3's aggregate column. Step 3 = 0.23 + 799/89.0 = 9.2 s. (MI355X row: TTFT 0.20 s at c=1 and 0.28 s at c=8; Step 1 = 0.20 + 199/65.1 = 3.3 s, Step 2 = 0.28 + 499/45.5 = 11.3 s, Step 3 = 0.20 + 799/65.1 = 12.5 s.)

**Printing is never the constraint.** Every row prints at 48–193 words/s against a human reading speed of ~4–5. Past ~10 words/s more per-user tok/s is imperceptible — it only shortens the tail before the reader can start scrolling. **What the user feels is the wait before the first word**, which is hidden token generation plus prefill.

**MTP is worth 4.7× on the whole turn** in the InferenceX B200 data (51.6 s → 10.9 s) — larger than any hardware gap in this table, and it shortens the hidden phases as well as the visible one (§6). On our MI355X ATOM, with real acceptance, it is worth **2.8×** (25.0 s → 8.9 s), and 2.2× on the short task (3.7 s → 1.7 s).

**Do not read the total column across the ours/InferenceX boundary.** InferenceX's TTFT is measured on 100k+ token agentic prompts with prefix caching on; ours on a 1024-token random prompt (§2). Compare ours-to-ours and InferenceX-to-InferenceX. The act step also assumes the 8 calls are independent — if each needs the previous one's result they run serially and that phase takes ~8× longer.

### A short task

The turn above is a heavy one. Most agent traffic is lighter — a brief plan, small tool payloads, a couple of sentences back. Same three steps and same measured rates, but **plan 30 tok → 8 tool calls of 70 tok → answer 100 tok** (~75 words):

| System | Plan (30 tok) | 8 tool calls (70 tok ea) | Answer (100 tok) | **Whole turn** | First word at | Print rate |
|---|---:|---:|---:|---:|---:|---:|
| **Ours** 16× B200 (no MTP) | 0.6 s | 1.3 s | 1.3 s | **3.2 s** | 2.1 s | 67 words/s |
| **Ours** 8× MI355X vLLM recipe 1 (+MTP) | 0.6 s | 1.8 s | 1.7 s | **4.2 s** | 2.6 s | 49 words/s |
| **Ours** 8× MI355X ATOM (no MTP) | 0.6 s | 1.4 s | 1.7 s | **3.7 s** | 2.2 s | 48 words/s |
| **Ours** 8× MI355X ATOM +MTP | 0.3 s | 0.8 s | 0.6 s | **1.7 s** | 1.3 s | 193 words/s |
| **InferenceX** 16× B200 (no MTP) | 4.9 s | 10.0 s | 5.7 s | **20.6 s** | 19.4 s | 61 words/s |
| **InferenceX** 16× B200 +MTP | 1.3 s | 1.3 s | 1.6 s | **4.2 s** | 3.7 s | 166 words/s |
| **InferenceX** 8× MI355X ATOM +MTP | 1.0 s | 1.9 s | 1.6 s | **4.5 s** | 3.7 s | 95 words/s |
| **InferenceX** 8× MI355X vLLM +MTP | 1.8 s | 3.3 s | 2.6 s | **7.6 s** | 6.5 s | 63 words/s |

**Prefill stops being negligible.** The B200 turn drops from 19.6 s to **3.2 s**, and the three prefills (0.69 s of TTFT) go from 4% of the heavy turn to **22%** of this one. Generation still dominates at this size, but the margin narrows — and on realistic agent prompts of 100k+ tokens, where TTFT is seconds rather than 0.23 s (§2), it would invert entirely.

## 6. MTP — what it is, and how to enable it

### What "MTP" is — and it is software, not silicon

**Short answer: yes — MTP is an inference *software* technique, not a hardware capability.** It says nothing about how fast the GPU is. Any vendor's hardware can run it once the software stack supports it. So the MTP columns above measure **software maturity, not silicon**, and must not be read as a hardware comparison.

**MTP = Multi-Token Prediction**, a form of **speculative decoding**:

- **Without it:** one forward pass = **one** token. Every activated expert's weights are read from HBM to emit that single token (~2 GB per GPU at c=1). The weight read is the cost, not the arithmetic.
- **With it:** a small draft model proposes the next **N** tokens; the full model **verifies all N in one pass** — one weight read instead of N. Correct guesses are kept, the first wrong one and everything after is redone.
- **Output is bit-identical** to normal decoding. Pure speedup, no quality tradeoff.

**Why it helps so much here:** decode is *latency-bound, not bandwidth-bound* (`../../b200-kimi/results/kimi-k3-base-b200.md` §3). Verifying N tokens at once makes each expert GEMM **N wide instead of 1** — the same widening `max-num-seqs` buys, but **without needing other users**. That is why it is the only lever that helps a single user.

Measured gain in the InferenceX B200 data (MTP vs no-MTP, same hardware):

| | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| B200 per-user tok/s, **no MTP** | 81.9 | 21.1 | 3.8 |
| B200 per-user tok/s, **MTP** | 221.7 | 154.1 | 46.5 |
| ratio | **2.7×** | **7.3×** | **12.2×** |

> **Note the trend is the opposite of what simple theory predicts.** Speculative decoding should matter *most* at low concurrency and fade as batching widens the GEMMs on its own — yet here the ratio grows from 2.7× to 12.2×. The likely cause is that the ratio is not a clean MTP-only A/B: the no-MTP column collapses steeply (81.9 → 3.8) under long-context agentic traces, and the InferenceX MTP rows come from a different recipe family than the `agg-b200-tp8pp2-agentic.yaml` config cross-checked in §1. Treat the **c=1 figure (2.7×) as the trustworthy one** and the high-concurrency ratios as confounded.

Measured gain in our MI355X ATOM data (MTP vs no MTP, same hardware, 1024/1024, real acceptance):

| | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| MI355X ATOM per-user tok/s, **no MTP** | 64.0 | 56.4 | 35.3 |
| MI355X ATOM per-user tok/s, **MTP** | 256.8 | 112.2 | 61.7 |
| ratio | **4.0×** | **2.0×** | **1.7×** |

Ceiling is draft length × acceptance rate — never the full N, since not every guess is accepted. In our MI355X vLLM recipe 1 run the measured acceptance length is 1.30–1.66 tokens per step at draft length 3–7 (`node6101/kimi-recipe/kimi-k3-recipe.md`, "accept len").

**One fact that matters for reading the table:** Kimi-K3 has no built-in MTP head (`num_nextn_predict_layers = 0`) — the gain comes from **DSpark**, a separate speculator model, downloaded and run alongside it. "MTP" names the technique, not a checkpoint feature — the how-to follows below, for both platforms.

### How to enable it — B200 and MI355X

Spec decoding does *not* require single-node TP. The upstream vLLM recipe gates DSpark off `multi_node_tp_pp` — a **recipe default, not an engine limit**. SemiAnalysis's own B200 recipe runs DSpark with `pipeline-parallel-size: 2`. **MTP is already built into our B200 image** (`vllm/vllm-openai:kimi-k3`) — the `dspark` method, `KimiK3MTP`, and `TOKENSPEED_MLA` are all present. No new package.

| Step | MI355X (our vLLM recipe 1) | MI355X (InferenceX) | B200 (InferenceX) | MI355X (our ATOM) |
|---|---|---|---| --- |
| 1. Speculator | `Inferact/Kimi-K3-DSpark`, mounted as `/draft` | `Inferact/Kimi-K3-DSpark` | `Inferact/Kimi-K3-DSpark` | `Inferact/Kimi-K3-DSpark` |
| 2. `pard_token` shim | not needed | not needed | **required** | not needed |
| 3. Extra server flags | none up to 14 users | none | 3 flags, below | `--method dspark --draft-model --num-speculative-tokens` |
| 4. Attention backend in the config | `ROCM_AITER_MLA` | `TRITON_MLA` | `TOKENSPEED_MLA` | ATOM default |
| 5. Draft length | 7, 5, 4, 3 as load rises (vLLM recipe) | 2 (SemiAnalysis recipe) | 7 (SemiAnalysis recipe) | 6 at 1 user, 2 above (InferenceX ATOM recipe) |

**Why MI355X is simpler:** one node holds the whole model, so the speculator drops straight into the existing TP8 launch — no shim, no extra flags.

**Why B200 needs more:** PP2 splits the model across 2 nodes, so the speculator's draft/verify state has to stay consistent across the pipeline stage boundary. That needs decode context parallelism plus a config shim (the checkpoint's `mask_token_id` must be aliased to `pard_token` in a local copy of `config.json`, or the speculative-config fails to load) and three extra server flags:

```
--decode-context-parallel-size 8
--dcp-comm-backend a2a
--attention-backend TOKENSPEED_MLA
```

`--speculative-config` JSON (same shape on all three, `attention_backend` and `num_speculative_tokens` differ as in the table above). Our MI355X vLLM recipe 1 at 1 user:

```json
{"model": "/draft", "num_speculative_tokens": 7, "method": "dspark",
 "attention_backend": "ROCM_AITER_MLA", "kv_cache_dtype": "fp8",
 "draft_sample_method": "probabilistic", "rejection_sample_method": "block"}
```

**On MI355X it is on in our vLLM recipe 1** (up to 14 users) and in our ATOM + MTP run (draft length 6 at 1 user, 2 above). **On B200 it was tried (job 26092372, TP8 × PP2, DCP 1) but the server did not start:** `(APIServer pid=3219376) NotImplementedError: Pipeline parallelism is not supported for this model. Supported models implement the `SupportsPP` interface.`. Given the ~2.7× per-user gain at c=1 measured above, this is the single highest-value follow-up available on B200.

## Source data

- Ours: [mi355x-vs-b200.md](mi355x-vs-b200.md) §5; B200 `../../b200-kimi/results/kimi-k3-base-b200.md`; MI355X `node6101/kimi-recipe/kimi-k3-recipe.{md,csv}` (script `../atom/run_kimi_recipe.sh`).
- InferenceX: `../../b200-kimi/semianalysis-ref/` (`SOURCE.md`, `inferencex-b200-mi355x.csv`, `agg-b200-tp8pp2-agentic.yaml`, `kimik3_fp4_mi355x_mtp.sh`, `kimik3_fp4_mi355x_atom_mtp.sh`).
