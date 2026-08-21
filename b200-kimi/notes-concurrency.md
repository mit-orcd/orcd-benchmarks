# Single-user speed vs concurrency — Kimi-K3 on 2 × 8 B200

Notes only. No jobs were run to produce this; every number is read from the completed
sweep in `results/kimi-k3-base-b200.csv` (job 20916742) and the MI355X baseline it is
compared against.

---

## 1. Is "one user, one request" the same as concurrency 1?

**Yes.** `--max-concurrency 1` means exactly one request in flight, which is the
single-user case.

But **total tok/s is the wrong column to read for that user.** The right metric is
**per-user token rate = 1000 / TPOT**, where TPOT is the median time per output token.
At c=1 those happen to nearly coincide (one user owns the whole engine); at higher
concurrency they diverge sharply, because the aggregate is split across users.

| Conc | Total tok/s | TPOT med (ms) | **Per-user tok/s** | What one user experiences |
|---:|---:|---:|---:|---|
| **1** | 86.7 | 11.24 | **89.0** | fastest possible |
| 2 | 165.2 | 11.85 | 84.4 | −5% |
| 4 | 280.8 | 13.62 | 73.4 | −18% |
| 8 | 475.3 | 15.41 | 64.9 | −27% |
| 16 | 801.8 | 18.65 | 53.6 | −40% |
| 32 | 1,209.3 | 25.28 | 39.6 | −56% |
| **64** | **1,696.4** | 35.89 | **27.9** | **−69%** |

Add TTFT (~226 ms at c=1) for the time to the *first* token; the table above is the
steady-state streaming rate after that.

---

## 2. The tension nobody should miss

**Total throughput and single-user speed move in opposite directions.**

Going c=1 → c=64 buys **19.6× aggregate throughput** and costs **3.2× per-user speed**
(89.0 → 27.9 tok/s). Both are real. Which one matters depends entirely on what is being
served:

| If you are optimising for… | Target | Read this column |
|---|---|---|
| A single interactive user, a latency SLO, one long agentic session | **run at low concurrency** | per-user tok/s |
| Serving many users at once, cost per token, GPU utilisation | **run at high concurrency** | total tok/s |

> **This is the crucial caveat on §3.2 of the main report.** Every one of the four
> improvement levers there raises *aggregate* throughput — and three of them do it by
> increasing batch size, which makes single-user speed **worse**, not better. Raising
> `--max-num-seqs` from 64 to 512 is the right call for a busy server and the wrong call
> for one user waiting on one answer.

---

## 3. B200 vs MI355X for a single user — B200 wins clearly

This is the one comparison where B200 is unambiguously ahead, and the aggregate-
throughput framing in §6 of the main report hides it completely.

| Conc | MI355X per-user tok/s | B200 per-user tok/s | **B200 / MI355X** |
|---:|---:|---:|---:|
| **1** | 46.6 | **89.0** | **1.91×** |
| 2 | 44.3 | 84.4 | 1.91× |
| 4 | 40.0 | 73.4 | 1.83× |
| 8 | 37.0 | 64.9 | 1.75× |
| 16 | 32.0 | 53.6 | 1.67× |
| 32 | 26.4 | 39.6 | 1.50× |
| 64 | 20.0 | 27.9 | 1.39× |

**B200 is faster per user at every concurrency, and nearly 2× faster for a single user** —
despite needing twice the GPUs and losing on tok/s *per GPU* (0.67×) and *per node*
(0.68×).

The two facts are not in conflict; they answer different questions:

- *"How many tokens can this hardware produce per second per dollar of silicon?"* →
  MI355X, by 1.48× per GPU.
- *"How fast does one user's answer stream?"* → B200, by 1.91×.

B200's per-token latency advantage is what wins the second question, and it survives
even the PP2 pipeline penalty.

---

## 4. Why none of the four §3.2 levers help at concurrency 1

All four work by putting **more tokens into each expert GEMM**. At c=1 there is exactly
one token, so there is nothing to widen:

| Lever | Effect at c=1 | Why |
|---|---|---|
| 1. Raise `max_num_seqs` | **none** | Raises the *cap*, but with one user the batch is still 1. The cap was never the binding constraint. |
| 3. Expert parallelism | marginal | 1 token still routes to only 16 experts; the GEMM stays a GEMV. |
| 4. P/D disaggregation | none | Nothing to disaggregate with a single request. |
| 2. **Speculative decoding** | **the one that would work** | Verifies ~8 tokens per weight read ⇒ one user gets an 8-wide GEMM. **Blocked here** (§5). |

At c=1 each GPU reads ~2 GB of weights to produce a single token: 11.24 ms TPOT, HBM at
roughly 1–2% of peak. That is pure latency. Batching cannot fix it because there is
nothing to batch.

---

## 5. What would actually improve single-user speed

Ranked by expected effect:

1. **Speculative decoding / MTP — the only real answer.** Verifies several tokens per
   weight read, so a single user gets a wide GEMM without any other users present. This
   is the classic single-stream latency fix, and it is precisely what this deployment
   cannot use: the vLLM recipe gates DSpark off `multi_node_tp_pp`
   (vllm-project/vllm#50098), and PP is **mandatory** on B200 because the 1561 GB
   checkpoint does not fit one node's 1538 GB. Native **MTP** is the workaround worth
   investigating, since it belongs to the model rather than to the DSpark path.

2. **Eliminate PP.** Every token crosses the inter-node boundary once per decode step,
   and at c=1 there is no batch to hide that behind — the pipeline bubble is at its worst
   exactly in the single-user case. Requires hardware where the model fits in one node
   (MI355X at 2304 GB/node, or B300 at 2144 GB). Not achievable on B200.

3. **Lower TP.** TP8 issues 2 all-reduces per layer — 186 per token at 14 KB each,
   firmly latency-dominated. With no batch to amortise them, they are a meaningful share
   of an 11.24 ms step. Lower TP means fewer collectives, but needs more memory per GPU,
   which B200 does not have here.

4. **CUDA graphs / launch-overhead reduction.** Fixed per-kernel cost matters most when
   there is no batch to spread it over. Already partly captured; worth profiling before
   assuming it is exhausted.

**The honest summary:** the two fixes that would genuinely help a single user —
speculative decoding and dropping PP — are *both* unavailable on B200 for the same
underlying reason: the model does not fit in one node. That is a distinct cost from the
throughput story, and it is the one place where the two-node requirement bites the user
experience rather than the utilisation number.

Even so, B200 still delivers **1.91× MI355X's single-user speed** — it starts from a
strong enough per-token latency that it wins anyway.

---

## 6. Practical recommendation

- **Serving one user / latency-sensitive:** run at low concurrency and leave
  `--max-num-seqs` alone. Expect ~89 tok/s per user. Do **not** apply the §3.2 levers —
  they trade away the thing being optimised.
- **Serving many users / throughput-sensitive:** apply §3.2 lever 1 (raise
  `--max-num-seqs`), accept ~28 tok/s per user at c=64, and see
  `kimi-k3-improve-b200.md` for measured gains.
- **Needing both:** that is what speculative decoding / MTP exists for, and it is the
  single highest-value thing to unblock on this platform.

---

## Source data

| What | Where |
|---|---|
| B200 sweep (7 points, TP8 × PP2) | `results/kimi-k3-base-b200.csv`, job 20916742 |
| MI355X baseline (ATOM, TP8, 1 node) | `../amd-benchmarks/amd-cloud/logs/atom/sweep_20260814_164903` |
| Full analysis | `results/kimi-k3-base-b200.md` |
| Improvement levers | `results/kimi-k3-improve-b200.md` |

Per-user tok/s is derived as `1000 / median_tpot_ms`; all TPOT and throughput values are
measured, not modelled.
