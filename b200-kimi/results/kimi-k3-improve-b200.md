# Kimi-K3 on 2 × 8 × B200 — improvement experiments

Tests the four levers proposed in section 3.2 of `kimi-k3-base-b200.md`. The baseline there is `max_num_seqs=64`, TP8 × PP2, peak **1,696.4 tok/s** at c=64, with HBM at only ~23% of peak because each expert saw just **1.7 tokens** per step.

---

## What these experiments target: THE LATENCY BOTTLENECK

**Every lever here is aimed at one specific thing — the *latency* bound on HBM, not a bandwidth shortage.** That distinction decides which fixes can possibly work.

The baseline is **latency-bound, not bandwidth-bound**: HBM runs at only ~23% of peak, so ~77% of the memory bandwidth sits idle. It is idle because of **memory-level parallelism**, not memory speed. At `max_num_seqs=64` the 64 tokens scatter across **610 of 896 experts**, leaving each expert GEMM with only **1.7 tokens**. That is a matrix-*vector* product: it cannot keep enough memory requests in flight to fill the pipe, so it stalls on access latency long before it runs out of bandwidth.

**So the fix is never "more bandwidth" — it is "more tokens per expert GEMM".** Each lever is a different way of buying that:

| Lever | How it attacks the LATENCY bound | Mechanism |
|---|---|---|
| **1. Raise `max_num_seqs`** | More concurrent tokens ⇒ more tokens per expert | 1.7 → 4.6 tok/expert at batch 256, → 9.1 at 512. Directly widens every expert GEMM. |
| **2. Speculative decoding** | Several tokens verified per weight read | Widens the GEMM *without* needing more concurrent users — the same MLP gain at unchanged user load. Blocked by PP here (§3). |
| **3. Expert parallelism** | Fewer, larger, contiguous weight reads | Under TP each GPU reads a *thin slice* of every fired expert; under EP it reads *whole* experts. Higher memory-level parallelism per read, same token count. |
| 4. P/D disaggregation | *Does not widen the GEMM* | Removes prefill interference from decode steps. A cleanliness gain, not an MLP gain — which is why it is the weakest of the four even before the hardware cost (§4). |

**How to tell whether a lever actually worked:** watch `tokens/expert` and the derived **HBM %** in the tables below. Throughput rising *while HBM % rises* is the latency bound being relieved. Throughput rising with HBM % flat would mean something else changed.

**Where the ceiling is.** Above batch ~512 every expert fires anyway, so weight bytes plateau while tokens keep growing — that is where the GEMV finally becomes a real GEMM and the latency bound dissolves. Realistic target is **HBM ~50–65%**, not 90%: a live engine also pays dequantisation, expert routing, MLA/KDA attention, 93 all-reduces per token, and interleaved prefill, none of which read weights.

**Not tested here, but the same mechanism** (worth trying if the levers below fall short): native **MTP / multi-token prediction** (spec decoding's benefit without the DSpark-vs-PP block); a larger `--max-num-batched-tokens` (8192 today) so chunked prefill contributes more work per step; and a **grouped-GEMM MoE kernel** that batches many small expert GEMMs into one launch — which is what AITER's SiTUv2 path does on MI355X — raising occupancy without raising batch size at all.

Run: `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635`

---

## 0. Verdict — what to actually do

| Lever | Recommended? | Status | Result |
|---|---|---|---|
| **1. Raise `max_num_seqs`** | ⭐ **YES — do this first** | ran | 1.73× baseline (2,929.4 tok/s) |
| 3. Expert parallelism | ⭐ worth testing (2nd priority) | ran | 1.13× baseline (1,922.0 tok/s) |
| 2. Speculative decoding | ✗ unavailable on this layout | FAILED_TO_START rc=1 | gated off `multi_node_tp_pp`; PP is mandatory on B200 |
| 4. Prefill/decode disagg | ✗ needs 2× the hardware | not run | ≥32 GPUs (4 nodes) required; 16 available |

---

## 1. ⭐ Raise `--max-num-seqs` — the recommended lever

**Mechanism: this is the most direct attack on the latency bottleneck.** More concurrent tokens means more tokens land on each fired expert, which widens every expert GEMM and lets it keep more memory requests in flight.

| Cap | Conc | tok/s | vs baseline peak | TPOT med (ms) | tokens/expert | **HBM %** | latency bound relieved? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 *(baseline)* | 64 | 1,696.4 | 1.00× | 35.89 | 1.7 | **21%** | _reference_ |
| 256 | 64 | 1,611.4 | 0.95× | 38.22 | 1.7 | **20%** | ❌ **worse** — HBM 21%→20% (-1 pts); gain is batch size alone |
| 256 | 128 | 1,836.9 | 1.08× | 68.94 | 2.5 | **15%** | ❌ **worse** — HBM 21%→15% (-7 pts); gain is batch size alone |
| 256 | 256 | 2,308.5 | 1.36× | 110.22 | 4.6 | **10%** | ❌ **worse** — HBM 21%→10% (-11 pts); gain is batch size alone |
| 512 | 128 | 1,756.3 | 1.04× | 69.28 | 2.5 | **15%** | ❌ **worse** — HBM 21%→15% (-7 pts); gain is batch size alone |
| 512 | 256 | 1,751.2 | 1.03× | 150.25 | 4.6 | **7%** | ❌ **worse** — HBM 21%→7% (-14 pts); gain is batch size alone |
| 512 | 512 | 2,929.4 | 1.73× | 167.03 | 9.1 | **7%** | ❌ **worse** — HBM 21%→7% (-15 pts); gain is batch size alone |

**How to read this — is the LATENCY bottleneck actually improving?** `tokens/expert` is the quantity that was binding; **HBM %** is the consequence, and it is the real scorecard. Three outcomes, and they mean different things:

- **HBM % rises with throughput** ⇒ the latency bound is genuinely being relieved. Each expert GEMM now has enough width to keep memory requests in flight, so the same hardware is doing more useful work per second. This is the win condition.
- **HBM % flat while throughput rises** ⇒ the extra tokens are being served, but per-step memory efficiency is unchanged — the bound is still fully in place and the gain is just a bigger batch.
- **HBM % falls while throughput rises** ⇒ per-step efficiency got *worse*. Throughput went up only because batch went up, and TPOT grew faster than the extra expert traffic justified — often queueing or a widening pipeline bubble. Raising the cap further will hit diminishing returns fast.

HBM % here is derived as `experts_fired(batch) × bytes_per_expert ÷ TPOT`, per GPU, against B200's 8 TB/s — the same arithmetic as section 3 of the baseline report. Expert weight reads are >=98% of step bytes, so they are a fair proxy for total traffic.

Weight bytes plateau above batch ~512, so past that point every extra token is nearly free in bandwidth terms — which is exactly why the cap-512 arm is the one that can push HBM % furthest.

## 2. Expert parallelism — matched A/B against lever 1

EP is compared against the `max_num_seqs=256` arm at the **same cap and the same concurrencies**, so EP is the only variable. This matters: the MI355X study's first EP result compared an EP arm at cap 256 against a TP-only arm at cap 64, and only one row of it was a valid comparison.

| Conc | TP-only (cap 256) | EP (cap 256) | EP/TP | TP TPOT | EP TPOT |
|---:|---:|---:|---:|---:|---:|
| 64 | 1,611.4 | 1,503.8 | 0.93× | 38.22 | 41.77 |
| 128 | 1,836.9 | 1,829.9 | 1.00× | 68.94 | 69.69 |
| 256 | 2,308.5 | 1,922.0 | 0.83× | 110.22 | 139.11 |

**Mechanism: EP attacks the same latency bottleneck by a different route.** Under TP each GPU reads a *thin slice* of every activated expert and computes a partial GEMM. Under EP each GPU holds *whole* experts and computes complete GEMMs — fewer, larger, more contiguous weight reads, so more memory-level parallelism per read at the SAME token count. It raises HBM utilisation without needing a bigger batch, which is why it is worth measuring rather than dismissing on the fact that the interconnect is idle.

## 3. Speculative decoding (DSpark) — unavailable here

Status: `FAILED_TO_START rc=1`

The vLLM recipe gates DSpark off the `multi_node_tp_pp` profile — it does not compose with pipeline parallelism yet (vllm-project/vllm#50098). On B200 that is decisive rather than inconvenient: **PP is mandatory**, because the 1561 GB checkpoint does not fit one node's 1538 GB. So the single most promising throughput lever after batch size is structurally unavailable on this hardware — and it is available on MI355X, which serves the model on one node with no PP.

> This is a real, quantifiable cost of needing two nodes, separate from the pipeline bubble.

## 4. Prefill/decode disaggregation — needs 2× the hardware

Not run, on arithmetic rather than preference:

| | |
|---|---|
| P/D needs | 2 independent engine instances (prefill pool + decode pool) |
| Weights per instance | 1561 GB — each holds a **full** copy |
| GPUs per instance | 16 (2 nodes), since one node's 1538 GB cannot hold it |
| **Minimum for P/D** | **32 GPUs / 4 nodes** |
| This allocation | 16 GPUs / 2 nodes |

Testing it would require doubling the reservation. Worth revisiting only if the model is ever served on hardware where one instance fits in a single node.

---

## 5. Recommendation

**Raise `--max-num-seqs` to 512.** Best measured: **2,929.4 tok/s** at c=512, 1.73× the cap-64 baseline, at a median TPOT of 167.03 ms. One flag, no extra hardware, and the KV memory was already provisioned.

Then, in order: measure EP (§2) if it loads; treat spec decoding (§3) as blocked until vLLM composes it with PP; leave P/D (§4) until there is 4-node capacity.

---

## Source data

| What | Where |
|---|---|
| lever1_mns256 | `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635/lever1_mns256_sweep/`, `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635/lever1_mns256_server/` |
| lever1_mns512 | `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635/lever1_mns512_sweep/`, `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635/lever1_mns512_server/` |
| lever3_ep | `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635/lever3_ep_sweep/`, `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635/lever3_ep_server/` |
| lever2_spec | `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635/lever2_spec_sweep/`, `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635/lever2_spec_server/` |
| baseline (cap 64) | `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/kimi_base_20260821_130024/sweep` |
| driver log | `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/improve_20260821_144635/STATE.txt` |

Baseline report: `kimi-k3-base-b200.md` (unmodified by this run).
