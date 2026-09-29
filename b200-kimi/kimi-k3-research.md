# Kimi-K3 as an HPC research subject — landscape and extensions

Two things, kept separate:

- **§1** — what other people have already published or are actively publishing, on
  (a) benchmarking Kimi-K3 and (b) using models like it to do HPC research.
- **§2** — extended work this site could do next, argued from the measurements
  already in `results/kimi-k3-base-b200.md` and `results/kimi-k3-improve-b200.md`.

Everything in §1 is cited. Everything in §2 is anchored to a number we measured, and
says plainly what is speculative.

---

## 1. What has been done, and by whom

### 1.1 The model's own system papers, and day-0 engine work

**Moonshot's technical report** ([arXiv:2607.24653](https://arxiv.org/pdf/2607.24653))
is the primary source for the architecture we re-derived from `config.json`: 2.8 T
parameters, ~104 B activated, hybrid **Kimi Delta Attention (KDA)** linear attention
interleaved with full attention at roughly 3:1, Attention Residuals, Stable LatentMoE,
896 routed experts with 16 active, 1 M context, native vision. They claim ~2.5×
better scaling efficiency than Kimi K2.

**vLLM published two rounds of engine work**, and this is the most directly relevant
external benchmarking to ours:

| | Day-0 preview / launch | Optimization round |
|---|---|---|
| Posts | [preview, 2026-07-22](https://vllm.ai/blog/2026-07-22-kimi-k3-preview), [launch, 2026-07-27](https://vllm.ai/blog/2026-07-27-k3) | [2026-09-13](https://vllm.ai/blog/2026-09-13-kimi-k3-performance-optimization) |
| Hardware | Blackwell | **B300 node, TP8** (single node — fits) |
| Workload | — | 8K in / 1K out, DSpark speculation, 8 draft tokens |
| Headline | functional day-0 support | **2.2–2.8× throughput, TTFT 72–85% lower, latency 56–60% lower** at c=1/4/16 |

Their seven named optimizations are worth reading against our §3 bottleneck finding,
because several attack exactly the mechanism we identified: adaptive scheduling budget
(TTFT −55–65%, throughput +41.5%), internal KDA prefix checkpoints, zero-copy mixed
KDA batches (+5.2–7.7% at c=4/16), deferred MXFP4 finalization (~5% end-to-end),
**ReplaySSM** (reconstructs KDA state instead of storing it, +10.97% effective cache
under TP), prefill/decode disaggregation, and **decode context parallelism** (shards
MLA KV along sequence, query-exchange latency −10.4–29.9%). At c=1 they move
83.3 → 183.3 tok/s.

Note the confound for us: that is a **B300** node, single-node, TP8, no PP. Their
baseline does not pay the two-node cost that dominates our §5.

**LMCache** publishes a [Kimi-K3 recipe](https://docs.lmcache.ai/recipes/kimi_k3.html)
and a [2026-04 architecture post](https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x/)
claiming ~13× TTFT reduction and >10× p99 tail improvement on MoE workloads through
unified KV-cache management. Untested against a KDA hybrid in our setting.

### 1.2 Independent third-party benchmarking of Kimi-K3

This is a crowded space, and our report already cross-checks against the largest of
them.

**SemiAnalysis InferenceX** ([dashboard](https://inferencex.semianalysis.com/inference/kimi-k3),
[commentary](https://newsletter.semianalysis.com/p/kimi-k3-the-manos-the-mythos-the))
— the reference we use in §7. Public, reproducible GitHub Actions runs, sweeping
hardware × framework × speculation method × concurrency, with an agentic-trace
workload rather than fixed ISL/OSL. Their B200 arms are 16 GPUs, TP8×PP2 — the same
layout as ours — which is why the comparison in §7.3/§7.4 is meaningful at all. Local
copy of their API dump and recipes is in `semianalysis-ref/`.

**AMD's ROCm blog** ([Benchmarking Kimi-K3 across vLLM, SGLang and ATOM on MI350X](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k3-mad/README.html))
— 8 × MI350X, ROCm 7.2.3, ISL 8192 / OSL 1024, concurrency 1→128, TP8, day-0
out-of-box configs. Their headline finding is convergence: at c=32 all three engines
land within 0.5% (vLLM 4,676, SGLang 4,694, ATOM 4,692 tok/s), with divergence at the
extremes attributed to untuned defaults rather than architecture. They explicitly
frame it as a day-0 snapshot, not a leaderboard.

**Wafer's cost-per-token study** ([post](https://www.wafer.ai/blog/kimi-k3-mi355x),
widely re-reported) is the one piece of external work that benchmarks a configuration
close to ours. At 1024 in / 400 out they report MI355X at 952 tok/s per node and
118 tok/s single-stream, against a **TP16 two-node B200** deployment at 498 tok/s
total — and conclude MI355X wins on tokens per dollar (48 tok/$, ~1.45× B300,
~6.9× B200). Treat the absolute numbers as vendor-adjacent and unaudited; a
[skeptical write-up](https://windowsforum.com/windows-news.4/kimi-k3-amd-mi355x-cost-win-unproven-8-gpu-hosting-validated.441481/)
argues the cost claim is unproven while the 8-GPU-hosting claim is solid.

> **Our 1,696.4 tok/s on the same 16 × B200 / two-node layout is 3.4× Wafer's 498.**
> Different OSL (1024 vs 400), different concurrency, possibly different `max_num_seqs`.
> That gap is large enough to be worth resolving and is item §2.1 below.

**Capability-side evaluation** is being done by many, and is *not* what we measure, but
it bounds what the serving numbers are worth: Moonshot report BrowseComp 90.4 at full
1 M context and 91.2 with compaction at 300 k;
[Semgrep](https://semgrep.dev/blog/2026/kimi-k3s-code-security-results-lack-precision/)
find code-security results strong on aggregate but weak on precision;
[IntuitionLabs](https://intuitionlabs.ai/articles/kimi-k3-long-context-evaluation)
note that as of September 2026 **no third party had published a needle-in-a-haystack
result at 500 k or 1 M tokens** — the headline context length is, in their reading,
independently unverified.

### 1.3 Academic and lab benchmarking of MoE / LLM inference on HPC

None of this work covers Kimi-K3 specifically. That is the gap.

| Work | Venue / link | What it does | Relation to ours |
|---|---|---|---|
| **MoE-Inference-Bench** | SC'25 workshops, [arXiv:2508.17467](https://arxiv.org/pdf/2508.17467) | Sweeps batch, seqlen, FFN dim, expert count, pruning, fused-MoE, spec decoding, quantization, parallelism — on H100, across Mixtral/DeepSeek/OLMoE/Qwen | **Closest methodological sibling.** Stops well short of 2.8 T and of hybrid linear attention; H100-era hardware |
| **LLM-Inference-Bench** | [arXiv:2411.00136](https://arxiv.org/pdf/2411.00136) | Inference benchmarking across AI accelerators, sub-1 B to 405 B | Establishes the multi-accelerator comparison format; no trillion-scale MoE |
| **ExaServe** | [arXiv:2609.10812](https://arxiv.org/pdf/2609.10812) | Large-scale LLM serving on ALCF Aurora — scheduler integration, MPI launch, node-local weight staging, declarative YAML→deployment. Near-linear to 256 nodes / 3,072 vLLM replicas, 27.1k req/s (3.8 M tok/s); streaming bottlenecks at ~4.7k req/s on a centralized proxy; 30-min init from an O(N²) Ray Serve control plane | **The HPC-systems framing we lack.** They scale *replicas*; we characterize *one* replica of a model too big for a node. Complementary |
| **Serving LLMs in HPC Clusters** | [arXiv:2507.00418](https://arxiv.org/pdf/2507.00418) | National Research Platform; Qualcomm Cloud AI 100 Ultra vs A100, 12 models, throughput-per-watt. 70 B on 1 QAic card vs 8 A100s: 148 W vs 2,983 W | The energy-per-token methodology we have not applied |
| **TokenPowerBench** | AAAI, [arXiv:2512.03024](https://arxiv.org/pdf/2512.03024) | Power consumption of LLM inference as a first-class benchmark axis | ditto |
| **Energy-per-token advocacy** | [arXiv:2603.20224](https://arxiv.org/pdf/2603.20224), [arXiv:2607.26571](https://arxiv.org/pdf/2607.26571) | Argues energy/token should be a standard metric alongside FLOPs and latency; analytical energy models for modern GPUs | Directly actionable on our sweep (§2.4) |

On the deployment side, [ALCF now runs a managed inference gateway](https://arxiv.org/pdf/2604.06217)
for open-weight models next to data and experiments — the institutional pattern our
work sits inside.

### 1.4 Using models like Kimi-K3 *to do* HPC research

The second half of the question. Active, and mostly benchmark-construction work:

- **PerfCodeBench** ([arXiv:2605.15222](https://arxiv.org/html/2605.15222)) — LLMs for
  system-level high-performance code optimization.
- **Comprehensive Evaluation of LLMs in HPC Code Performance Optimization**
  ([ICPP'25 workshops](https://dl.acm.org/doi/10.1145/3750720.3757280)) — HPC
  computational motifs, o1 / Claude-3.5 / Llama-3.2 / HPC-Coder.
- **PETScAgent-Bench** ([arXiv:2603.15976](https://arxiv.org/html/2603.15976)) — does
  AI-generated code use a production HPC library the way an expert would? Tool-augmented
  evaluator that compiles, runs and measures.
- **AInsteinBench** ([arXiv:2512.21373](https://arxiv.org/html/2512.21373)) — LLM agents
  as scientific-computing development agents inside real research software ecosystems.
- **ParaCodex** ([arXiv:2601.04327](https://arxiv.org/pdf/2601.04327)) — profiling-guided
  autonomous coding agent for parallel code generation and translation.
- **Parallelization capabilities of agentic LLMs**
  ([Springer](https://link.springer.com/chapter/10.1007/978-3-032-35248-4_2)) and
  **LLM & HPC: DeepSeek on HPC tasks**
  ([Springer](https://link.springer.com/chapter/10.1007/978-3-032-07612-0_48)).

Consistent finding across these: frontier models write readable, well-structured
parallel code but fail on correctness for hard problems and on library-specific
conventions even when the code compiles and runs.

**None of them evaluates Kimi-K3**, and none of them runs the model on the evaluating
institution's own hardware. That combination — self-hosted frontier open weights,
evaluated on HPC tasks, with the serving cost of each evaluation measured — is
unclaimed ground.

### 1.5 The gaps, stated plainly

1. No published trillion-scale-MoE inference characterization from an **academic HPC
   center** on its own hardware. The serving numbers all come from vendors, model
   authors, or commercial benchmarkers.
2. No **energy-per-token** figure for Kimi-K3 on any hardware.
3. No independent **long-context** (≥500 k) serving characterization — everyone
   benchmarks at 1 k–8 k input while the model advertises 1 M.
4. No study of the **two-node-vs-one-node** penalty for a model that straddles node HBM
   capacity, which is the single most interesting thing about this model on B200.
5. Kimi-K3 is absent from every HPC-agent benchmark, despite being the largest open
   weights available to run one.

---

## 2. Extended work, ranked

### What we already have

| Result | Where |
|---|---|
| 1→64 concurrency sweep, 16 × B200, TP8×PP2, ISL/OSL 1024/1024, `max_num_seqs=64` | `results/kimi-k3-base-b200.md` |
| Peak **1,696.4 tok/s**; HBM **~23%** of peak, compute 0.97%, NVLink 0.46%, IB 0.01% | §1, §3 |
| Bottleneck identified as **latency-bound, not bandwidth-bound**: 610 of 896 experts fire per layer at batch 64, leaving **1.7 tokens per expert** — a GEMV | §3.1, §3.2 |
| Exact KV arithmetic: 13,824 B/token, only the 24 MLA layers page KV, replicated not sharded across TP | §2 |
| Four improvement levers run: `max_num_seqs`→512 **2,929.4 tok/s (1.73×)**, EP **1,922.0 (1.13×)**, spec decoding failed to start, P/D needs 4 nodes | `results/kimi-k3-improve-b200.md` |
| B200 vs MI355X head-to-head and a five-subsection cross-check against InferenceX | §6, §7 |

Six directions follow, ordered by (value of the answer) ÷ (cost to get it).

---

### 2.1 Close the gap with Wafer's two-node B200 number — *cheapest, highest credibility return*

**The problem.** Wafer reports **498 tok/s** for TP16 two-node B200; we measure
**1,696.4** on the same layout. A 3.4× discrepancy between the only two public
measurements of this configuration is a result in itself, whichever way it resolves.

**Why it probably differs.** Three candidates, all testable: OSL (400 vs 1024, and
short OSL amortizes prefill over fewer decode steps), `max_num_seqs` (ours 64 —
theirs unstated), and concurrency at the reported point. Our own §3.3 shows TTFT
nearly flat while TPOT rises 3.2× across the sweep, so the ISL/OSL ratio moves the
aggregate number a lot.

**Do.** Re-run the existing sweep at ISL/OSL 1024/400 and at 8192/1024 (the ROCm blog's
shape, which also makes our numbers directly comparable to their MI350X engine
convergence result). Two sweeps, same harness, no new code.

**Produces.** An ISL/OSL sensitivity surface, plus either a reconciliation or a
documented contradiction with the only comparable public number.

---

### 2.2 Actually get MTP running on TP8×PP2 — *the largest single throughput lever left*

**Status is wrong in our own files.** `results/kimi-k3-improve-b200.md` §3 still says
spec decoding is "structurally unavailable" because vLLM gates DSpark off
`multi_node_tp_pp`. `notes.md` corrects this: SemiAnalysis run DSpark **on B200
TP8×PP2, 16 GPUs**, and their recipe is in `semianalysis-ref/`. What they do that we
do not:

1. A different speculator — `Inferact/Kimi-K3-DSpark`, `num_speculative_tokens: 7`.
2. A **compat shim** that injects `pard_token = mask_token_id` into the checkpoint's
   `config.json` (the Inferact checkpoint publishes the parallel-drafting token under
   the wrong name; without this the speculative config simply does not load — this,
   not PP, is the likely real reason the upstream recipe avoids the combination).
3. `decode-context-parallel-size: 8`, `dcp-comm-backend: a2a`,
   `attention-backend: TOKENSPEED_MLA`, plus Mooncake KV offload.

**Why it matters mechanically.** §3.2 says the fix for a latency bound is *more tokens
per expert GEMM*. MTP delivers exactly that without needing more concurrent users —
it verifies several tokens per weight read. Our §7.4 measures InferenceX's own
MTP-vs-no-spec gap at **2.7× at c=1** on B200.

**Do.** Download `Inferact/Kimi-K3-DSpark` (blocked today by `HF_HUB_OFFLINE=1`), apply
the shim, add the four server flags to `job-kimi-base.sh`, re-sweep.

**Then fix the two reports** that still carry the "structurally unavailable" framing —
`kimi-k3-improve-b200.md` §3 and `notes-concurrency.md`.

**Risk.** Real. Four interacting flags we have never run, on a layout the upstream
recipe deliberately avoids. Budget a debugging session, not a re-run.

---

### 2.3 The long-context sweep nobody has published

**The gap.** §1.5 item 3. Every public Kimi-K3 serving benchmark uses 1 k–8 k input.
The model advertises 1 M. Our own `max_model_len` is **16384** — we have not touched
the regime the model is sold on.

**Why this configuration is the right one to ask.** Two structural facts from our
report make the long-context question sharper here than anywhere else:

- KV is **13,824 B/token**, and only the 24 MLA layers page it (§2). The 69 KDA layers
  carry fixed-size recurrent state instead. So KV growth with context is ~3.9× slower
  than a same-depth full-attention model — that is the whole point of the hybrid, and
  it is directly measurable.
- **Prefix caching is disabled for correctness** (§5.3) because KDA recurrent state
  cannot be rebuilt from the paged MLA cache. In long-context agentic workloads with
  shared prefixes, this forfeits a large win that non-KDA models get for free. vLLM's
  **ReplaySSM** (§1.1) is aimed precisely at this; nobody has measured what it recovers.

**Do.** Raise `max_model_len` toward 256 k, sweep ISL at 16 k / 64 k / 256 k against
OSL 1024, and record TTFT, TPOT, KV-pool occupancy and HBM % at each. Our KV pool is
59.1 GiB/GPU and was **~2.0% used** — there is room.

**Produces.** The first published KV-scaling curve for a hybrid-linear-attention
frontier model, and a measured price for the disabled prefix cache.

---

### 2.4 Energy per token — *unclaimed, and cheap*

**The gap.** §1.5 item 2: no energy figure exists for Kimi-K3 on any hardware. Meanwhile
[arXiv:2603.20224](https://arxiv.org/pdf/2603.20224) argues energy-per-token belongs
beside FLOPs and latency as a standard metric, and
[TokenPowerBench](https://arxiv.org/pdf/2512.03024) and
[arXiv:2507.00418](https://arxiv.org/pdf/2507.00418) supply ready methodology.

**Why our run is unusually interesting.** We measured compute at **0.97%** of B200 peak
and HBM at ~23%. A chip drawing near-TDP to do 1% of its arithmetic is the definition
of an energy-inefficient operating point, and the **two-node requirement doubles the
GPU count for a model that fits on one MI355X node**. Joules per token is the metric
that prices that penalty in a way tokens-per-second cannot.

**Do.** Sample `nvidia-smi --query-gpu=power.draw` (or DCGM) during the existing sweep
stages. Report J/token against concurrency, and against MI355X's 8-GPU node.
No new runs needed if folded into any sweep above — this is instrumentation, not an
experiment.

**Caveat to state up front.** Node-level power (CPU, NICs, cooling) is not captured by
GPU counters; report GPU-only energy and say so.

---

### 2.5 The two-node penalty, decomposed

**The gap.** §1.5 item 4, and our own §6.7 admits it: *"Does not measure the pipeline
bubble, which needs per-stage profiling this run does not collect."*

**Why it is the most novel thing here.** Kimi-K3 on B200 is a natural experiment. The
checkpoint is **1561 GB** against a node's **1538 GB** — 23 GB short. That 1.5% miss
forces PP2, a second node, an InfiniBand hop in the per-token critical path, a pipeline
bubble, and (until §2.2) a broken speculative-decoding path. A B300 node
(8 × 268 GB = 2144 GB) would hold it outright. Nobody has quantified what a
1.5%-over-capacity model costs.

**Do.**

1. Per-stage profiling to size the bubble directly — the missing measurement.
2. If B300 or GB200 nodes ever become available here, the single-node arm turns this
   from an estimate into a controlled A/B.
3. Failing that, an EP-only arm at TP8×EP2 across the two nodes as an alternative
   16-GPU layout without a pipeline boundary, measured against the PP2 baseline.

**Produces.** A publishable statement of the form "crossing a node's HBM capacity by
1.5% costs X% of throughput and Y% of the speculative-decoding speedup," which is a
procurement-relevant result, not just a benchmark.

---

### 2.6 Use Kimi-K3 to do HPC work, and measure both halves

**The gap.** §1.5 item 5. Every HPC-agent benchmark in §1.4 calls a hosted API. None
runs the model on the institution's own hardware, and none reports what each evaluation
*cost* to serve.

**Why we are positioned for it.** §7.5 of our report already models an agent turn
end-to-end from measured TTFT and per-user tok/s — plan, fan out 8 parallel tool calls,
stream the answer — for both our systems and both of InferenceX's. That machinery
converts a serving sweep into an answer about agent latency. Point it at a real
workload and it becomes an evaluation.

**Do.** Run one of the existing HPC-agent benchmarks — PETScAgent-Bench or
AInsteinBench are the best-specified — against our locally served endpoint, and report
per-task **accuracy alongside GPU-seconds and joules**.

**Be honest about scope.** This is a substantially bigger project than §2.1–2.5: it
needs the instruct/agent-capable serving path, tool-calling wired up, and a harness we
do not currently have. List it as a direction, not a next sprint.

---

### Suggested order

| | Item | Cost | Why this order |
|---|---|---|---|
| 1 | §2.1 ISL/OSL sensitivity | 2 sweeps, no new code | Resolves a public 3.4× discrepancy |
| 2 | §2.4 Energy per token | Instrumentation only | Free if folded into §2.1; fills a documented gap |
| 3 | §2.2 MTP on PP2 | 1 debugging session | Largest throughput lever; also corrects two of our own files |
| 4 | §2.3 Long-context sweep | Config change + sweeps | Genuinely unpublished; KV headroom already exists |
| 5 | §2.5 Two-node penalty | Profiling; ideally new hardware | Most novel result, highest hardware dependency |
| 6 | §2.6 Agent evaluation | New harness | Different project |

---

## Sources

Model and engines —
[Kimi K3 technical report](https://arxiv.org/pdf/2607.24653) ·
[Kimi K3 on GitHub](https://github.com/MoonshotAI/Kimi-K3) ·
[Hugging Face](https://huggingface.co/moonshotai/Kimi-K3) ·
[vLLM preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview) ·
[vLLM day-0](https://vllm.ai/blog/2026-07-27-k3) ·
[vLLM optimization round](https://vllm.ai/blog/2026-09-13-kimi-k3-performance-optimization) ·
[LMCache recipe](https://docs.lmcache.ai/recipes/kimi_k3.html) ·
[LMCache MoE architecture](https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x/)

Third-party benchmarking —
[SemiAnalysis InferenceX](https://inferencex.semianalysis.com/inference/kimi-k3) ·
[SemiAnalysis commentary](https://newsletter.semianalysis.com/p/kimi-k3-the-manos-the-mythos-the) ·
[AMD ROCm: vLLM vs SGLang vs ATOM on MI350X](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k3-mad/README.html) ·
[Wafer: Kimi K3 on MI355X](https://www.wafer.ai/blog/kimi-k3-mi355x) ·
[Skeptical response](https://windowsforum.com/windows-news.4/kimi-k3-amd-mi355x-cost-win-unproven-8-gpu-hosting-validated.441481/) ·
[Semgrep code-security precision](https://semgrep.dev/blog/2026/kimi-k3s-code-security-results-lack-precision/) ·
[IntuitionLabs long-context evaluation](https://intuitionlabs.ai/articles/kimi-k3-long-context-evaluation)

HPC inference benchmarking —
[MoE-Inference-Bench](https://arxiv.org/pdf/2508.17467) ·
[LLM-Inference-Bench](https://arxiv.org/pdf/2411.00136) ·
[ExaServe](https://arxiv.org/pdf/2609.10812) ·
[Serving LLMs in HPC Clusters](https://arxiv.org/pdf/2507.00418) ·
[TokenPowerBench](https://arxiv.org/pdf/2512.03024) ·
[Energy-per-token advocacy](https://arxiv.org/pdf/2603.20224) ·
[Analytical energy estimation](https://arxiv.org/pdf/2607.26571) ·
[Open-weight models and sovereign AI](https://arxiv.org/pdf/2604.06217)

LLMs for HPC —
[PerfCodeBench](https://arxiv.org/html/2605.15222) ·
[LLMs in HPC code optimization](https://dl.acm.org/doi/10.1145/3750720.3757280) ·
[PETScAgent-Bench](https://arxiv.org/html/2603.15976) ·
[AInsteinBench](https://arxiv.org/html/2512.21373) ·
[ParaCodex](https://arxiv.org/pdf/2601.04327) ·
[Agentic parallelization capabilities](https://link.springer.com/chapter/10.1007/978-3-032-35248-4_2) ·
[DeepSeek on HPC tasks](https://link.springer.com/chapter/10.1007/978-3-032-07612-0_48)

Local measurements — `results/kimi-k3-base-b200.md`, `results/kimi-k3-improve-b200.md`,
`notes.md`, `semianalysis-ref/`.
