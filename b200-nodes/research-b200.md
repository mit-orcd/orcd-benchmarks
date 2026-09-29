# B200 benchmarking as HPC research — landscape and extensions

**Purpose.** This file sits alongside the measurement summaries in this
directory (`out-gpu-fryer/`, `out-nccl-1node/`, `out-nccl-2node/`,
`out-nccl-2node-sharp/`, `out-ibwrite/`, `output-megatron/`, `notes.md`,
`notes-aicr.md`) and asks a different question: **what would turn this
benchmarking work into research?**

Section 1 reviews what other groups have published or are actively working on.
Section 2 proposes extensions specific to what we have already measured here.

- Written: 2026-09-23
- Scope: NVIDIA B200 (Blackwell, sm100) in a multi-node HPC cluster context

---

# 1. What other researchers have done, and are doing

Six distinct lines of work. Our existing measurements already touch four of
them; two are untouched.

> *Provenance: compiled from a literature search on 2026-09-23. Headline numbers
> quoted below come from the papers' own abstracts/summaries; for several entries
> only title- and abstract-level detail was retrieved, so verify any figure
> against the source before citing it.*

## 1.1 Architecture microbenchmarking — the most crowded area

Two independent groups have published full microbenchmark dissections of
Blackwell, both released late 2025 and revised through 2026:

| Work | Focus |
|---|---|
| [Microbenchmarking NVIDIA's Blackwell Architecture](https://arxiv.org/abs/2512.02189) (Jarmusch et al., arXiv:2512.02189) | Memory subsystem, tensor-core pipeline, FP32/FP16/FP8/FP6/FP4 across B200 vs H200 |
| [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/pdf/2507.10789) (arXiv:2507.10789) | Independent dissection, same generation |

Headline results from the first: **1.85x ResNet-50** and **1.55x GPT-1.3B**
mixed-precision training throughput over H200, **~32-42% better energy
efficiency**, and a **58% reduction in memory-access latency on cache misses**.
They characterize the generation's new structures — 5th-gen tensor cores, tensor
memory (TMEM), the decompression engine, and the dual-die design.

**Implication for us: this area is saturated.** Single-GPU kernel-level
characterization of B200 is done, twice, by groups with more instrumentation
access than we have. Our gpu-fryer numbers (FP32 744-760, BF16 1437-1465, FP8
3949-4062 TFLOP/s per GPU) corroborate the published picture but add nothing new
at that level. **Our leverage is above the single GPU — at the node, fabric, and
cluster level, where almost none of this literature operates.**

## 1.2 Collective communication and the interconnect — active and unsettled

This is where the open questions are, and it is where most of our data already
sits.

- [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and
  Algorithms](https://www.alphaxiv.org/abs/2507.04786v2) (arXiv:2507.04786) —
  reverse-engineers NCCL's protocol/algorithm selection (Simple/LL/LL128 x
  Ring/Tree/CollNet/NVLS). Directly relevant to reading our per-collective
  results.
- [GPU-Initiated Networking for NCCL](https://arxiv.org/abs/2511.15076)
  (arXiv:2511.15076) — GIN: device-initiated communication inside NCCL's
  runtime, rather than CPU-proxy-driven. Targets exactly the small-message and
  irregular-pattern regime where our `alltoall` and `gather` collapse.
- [The Landscape of GPU-Centric Communication](https://arxiv.org/pdf/2409.09874)
  (arXiv:2409.09874) — survey framing the NVSHMEM / GIN / proxy-free direction.
- [Optimizing Allreduce Operations for Modern Heterogeneous Architectures with
  Multiple Processes per GPU](https://arxiv.org/pdf/2508.13397)
  (arXiv:2508.13397) — MPS-style multi-process-per-GPU allreduce.
- [EPIC: Abstraction and Polymorphism of In-Network Collectives on
  Ethernet](https://arxiv.org/pdf/2605.18683) (arXiv:2605.18683) — in-network
  reduction generalized beyond InfiniBand/SHARP.

On SHARP specifically, the consensus in the practitioner literature is that
CollNet/SHARP offloads all-reduce into the switch for near-zero GPU overhead and
is the preferred path **when the fabric supports it** — which, per `sharp.md`, is
precisely what ours does not currently do (no `sharp_am` aggregation manager).

**Implication for us:** our 2-node table has an `all_reduce` at 60% of the
fabric ceiling against `all_gather` at 92% — a clean, quantified statement of the
Ring two-pass penalty that SHARP exists to remove. We are sitting on a
well-instrumented *negative* control for the entire in-network-reduction
literature.

## 1.3 Low precision below FP8 — the Blackwell-specific story

FP4 (e2m1) and FP6 (e3m2/e2m3) are new in this generation: **FP4 throughput is
4x FP8**, FP6 close to FP8. NVIDIA's own [NVFP4
work](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
and the microbenchmark papers above cover the format mechanics — low precision
for operands, higher precision for accumulators in TMEM.

The interesting and **under-explored** thread is emulation of higher precision
from low-precision tensor cores for *scientific* (not ML) computation — using
FP8/FP6/FP4 pathways plus error-compensation to reach effective FP32/FP64
accuracy, which matters enormously because B200's native FP64 rate is weak
relative to its tensor-core rates.

## 1.4 Energy and power — methodologically active, B200 data thin

A dense 2026 literature on measurement methodology:

| Work | Contribution |
|---|---|
| [Wattchmen](https://arxiv.org/pdf/2603.26435) (arXiv:2603.26435) | High-fidelity GPU energy modeling; questions the accuracy of NVML/DCGM counters themselves |
| [PowerSensor3](https://arxiv.org/pdf/2504.17883) (arXiv:2504.17883) | Open-source external power measurement — ground truth against on-die telemetry |
| [Benchmark-driven Models for Energy Analysis and Attribution](https://dl.acm.org/doi/10.1145/3712285.3759815) (SC'25) | Attributes energy by functional unit (FPU / tensor core / ALU) and memory level |
| [Architectural Trade-offs in the Energy-Efficient Era](https://arxiv.org/html/2604.11391v1) (arXiv:2604.11391) | Power-capping study, H100 vs H200 |
| [The Energy Cost of Execution-Idle in GPU Clusters](https://arxiv.org/html/2604.04745v1) (arXiv:2604.04745) | Idle/allocated-but-unused energy — a scheduler-level concern |
| [Characterizing GPU Energy Usage in Exascale-Ready Portable Science Applications](https://link.springer.com/chapter/10.1007/978-3-032-07612-0_14) | Real science apps, not microbenchmarks |

One 2026 telemetry study spans 756 GPUs across generations but includes **only 8
B200s**. Blackwell energy data at cluster scale is genuinely scarce.

**Implication for us: this is our largest untouched dimension.** We have 80
B200s across 10 nodes and a stress benchmark (gpu-fryer) that already drives them
to a converged steady state per precision — the ideal substrate for a
TFLOP/J-per-precision study, and we collect none of it.

## 1.5 Cluster-scale variability, fail-slow, and straggler detection — the closest match to what we actually did

This is the line of work our AICR investigation belongs to, and it is very
active:

| Work | Contribution |
|---|---|
| [Quantifying Performance Variability in GPU Clusters](https://www.computer.org/csdl/journal/td/2026/06/11481967/2fHHzPybVuM) (IEEE TPDS, 2026) | Systematic variability characterization |
| [The Case of the Elusive Application Performance on GPU clusters](https://www.cs.umd.edu/~bhatele/pubs/pdf/2026/ipdps2026b.pdf) (IPDPS 2026) | Notes explicitly that GPU-cluster variability has *not* been analyzed as systematically as decades of CPU-HPC work |
| [ARGUS](https://arxiv.org/html/2606.20374) (arXiv:2606.20374) | Production tracing on 10,000+ GPUs; progressive fail-slow diagnosis down to straggler rank and kernel |
| [Guard](https://arxiv.org/pdf/2605.17879) (arXiv:2605.17879) | Node health management; cut run-to-run variance from 20% to 1% |
| [From Detection to Recovery](https://arxiv.org/pdf/2605.09370) (arXiv:2605.09370) | Operational analysis, 504-GPU pretraining |
| [Characterizing Production GPU Workloads](https://arxiv.org/pdf/2502.18680) (arXiv:2502.18680) | System-wide telemetry characterization |

The motivating statistics are stark: **59%** of 512-1024-GPU jobs in large
clusters hit fail-slow stragglers, with **~34.6%** average job-completion delay;
**42.5%** of jobs in production LLM clusters were straggler-affected, wasting
**10.4%** of GPU hours.

**Implication for us:** almost all of this work is *detection at scale from
telemetry* — statistical anomaly-finding across thousands of GPUs. What is
comparatively missing is **root-cause localization on a small cluster**: given
one node pair that is slow, which layer is at fault? That is exactly the
methodology we built and exercised twice (the IOMMU regression in `notes.md`, the
AICR PCIe-switch investigation in `aicr-handoff/`), and it is a genuine gap in
the published literature.

## 1.6 Traditional HPC benchmarks on Blackwell — surprisingly little public data

HPL/HPCG are supported on B200 through the [NVIDIA HPC-Benchmarks
container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks),
and vendor projections cite ~2.2x HPL for GB200 NVL4 over GH200 CG4. A notable
caveat in the same documentation: **B300 (sm103) has low native FP64 throughput**
and HPL is expected to perform poorly there — a warning that the Blackwell line
is diverging from FP64 simulation workloads.

Published, independent, per-site HPL/HPCG/science-app numbers for **B200** are
thin. For an HPC center whose users run simulation rather than LLM training,
"what does a B200 actually do for FP64 work?" is an unanswered and immediately
practical question.

Also relevant to methodology: [AI Benchmark Democratization and
Carpentry](https://arxiv.org/pdf/2512.11588) (arXiv:2512.11588) argues for
reproducible, site-runnable benchmark practice — the framing under which a
well-documented site benchmarking suite is itself a contribution.

## 1.7 Where our existing work already sits

| Area | Literature | Our coverage |
|---|---|---|
| Single-GPU architecture | Saturated (two full dissections) | Corroborating only |
| Collectives / interconnect | Active, unsettled | **Strong** — 10 collectives 1-node, 9 collectives 2-node, all pairs, vs derived HW ceiling |
| In-network reduction (SHARP) | Active | **Negative control** — measured, documented as unavailable |
| Sub-FP8 precision (FP4/FP6) | Active | **None** |
| Energy / power | Very active | **None** |
| Cluster variability / fail-slow | Very active, mostly at 10k scale | **Strong and differentiated** — two root-caused regressions |
| Traditional HPC (FP64, science apps) | Thin for B200 | **None** — Megatron only |

---

# 2. Extended work for B200, based on what we have measured

Ranked by value per unit effort. Each item states the observation in our data
that motivates it, what to run, and what would make the result worth writing up.

## 2.1 The two collectives that are 4-8x below the fabric — and why

**What our data shows.** In `out-nccl-2node/summary.md`, against a hardware
ceiling derived from this cluster (8 rails x 50 GB/s = 400 GB/s):

| Collective | % of ceiling | Effective rails (of 8) |
|---|---:|---:|
| sendrecv | 99% | 0.99 per pair |
| reduce_scatter / reduce / broadcast / all_gather | 92-94% | 7.3-7.5 |
| all_reduce | 60% | 4.80 |
| gather | 24% | 1.91 |
| **alltoall** | **12%** | **0.95** |

`alltoall` engaging **exactly one rail out of eight** is a precise, quantified
statement of an algorithmic failure, not a hardware one — and we proved it is
algorithmic: our fabric is ~1.9x the reference on `sendrecv`, yet `alltoall`
improved only 1.19x and `gather` only 1.05x. **A faster fabric barely helps a
collective that does not use it.**

**What to run.**
1. `NCCL_DEBUG=INFO` channel/rail mapping for `alltoall` vs `all_gather` — show
   directly which NICs are idle.
2. Sweep `NCCL_NCHANNELS`, `NCCL_MIN_NCHANNELS`, `NCCL_PXN_DISABLE`,
   `NCCL_CROSS_NIC`, and `NCCL_IB_QPS_PER_CONNECTION` to see whether *any*
   tuning recovers multi-rail alltoall.
3. Compare against **NVSHMEM** and, where available, NCCL's **GIN**
   (arXiv:2511.15076) on the identical pattern.
4. Add a **multi-node MoE-style all-to-all** at realistic message sizes (expert
   routing is the real-world consumer of this pattern).

**Why it is worth writing up.** The GIN and GPU-centric-communication papers
motivate their work on exactly this regime but rarely publish a clean per-rail
accounting on NDR-class hardware. "One rail out of eight, and here is the channel
map proving it" is a concrete, reproducible contribution.

## 2.2 Quantify what SHARP would buy — using the gap we already measured

**What our data shows.** `all_reduce` at 239.9 GB/s vs `all_gather` at 366.8
GB/s. A perfectly pipelined ring all-reduce would score the *same* as all_gather
under the busbw formula; ours reaches only **~65%** of it, against the
reference's ~78%, because ring fill/drain latency is fixed and does not shrink as
bandwidth grows. SHARP collapses the two passes into one in-switch reduction —
and `sharp.md` documents that it is unavailable here for a fabric-configuration
reason (no `sharp_am`), not a hardware one.

**What to run.**
1. **NVLS first — it may already work.** NVLink SHARP runs in the NVSwitch and
   needs no fabric-side aggregation manager. Test `NCCL_ALGO=NVLS` /
   `NVLSTree` on 1-node and 2-node all-reduce. If NVLS engages intra-node, we can
   decompose the two-pass penalty into its intra- and inter-node halves **today**,
   with no admin action.
2. Build the projection: measured ring all-reduce vs an upper bound from our own
   all_gather, framed as "what in-network reduction is worth on this fabric".
3. Keep the `sharp_am` request with the fabric admins live; re-run
   `job-nccl-2node-sharp-aicr.sh` the moment it appears.

**Why it matters.** A site-level, hardware-grounded answer to "is SHARP worth
enabling?" — with the number attached — is more useful to other HPC centers than
another paper asserting that it is.

## 2.3 Publish the fault-localization methodology — our most differentiated asset

**What our data shows.** We have root-caused **two** real inter-node GPUDirect
regressions, from opposite causes, with a consistent method:

| Case | Symptom | Root cause | Evidence |
|---|---|---|---|
| Engaging, 2026-07-13 | NIC reads from GPU 147.6 Gb/s (vs 395 line rate); NCCL sendrecv 12.7 GB/s | `iommu=pt intel_iommu=on` | `notes.md` — recovered to 395.5 Gb/s under `iommu=off` |
| AICR, 2026-08 | GPU RDMA **bidirectional** 27.2 GB/s/dir while unidirectional is healthy at 47.5 | Open; PCIe switch (Broadcom PEX890xx) is the surviving hypothesis after eliminating 7 others | `aicr-handoff/` |

And we documented a **methodological hazard** with a measured magnitude: binding
to a NODE-distance rail instead of the GPU's PXB partner gives **18.6 GB/s
instead of 49.4** — a **2.6x** error that mimics a hardware defect exactly, is
not recoverable by adding queue pairs (18.5 / 19.2 / 19.4 / 19.4 / 16.5 GB/s at
q = 1/2/4/8/16), and cost us a full round of wrong conclusions before we caught
it.

**What to build.** A portable **inter-node GDR acceptance test** that, in one
job, runs the ladder we converged on by hand:

```
rail affinity check  ->  host uni  ->  host bidir  ->  GPU uni  ->  GPU bidir
      ->  direction isolation  ->  concurrency scaling (1/2/4/8 pairs)
      ->  NCCL sendrecv cross-check
```

and emits a **decision table** mapping the observed pattern to the implicated
layer. We have already built most of the pieces: `run-engaging-check.sh`,
`job-ibwrite-rails.sh`, `job-ibwrite-concurrent.sh`, and
`analyze-engaging-check.py` (which already auto-detects the rail-affinity trap
and refuses to report invalid numbers).

**Why it is worth writing up.** The fail-slow literature (ARGUS, Guard) detects
anomalies statistically across 10k GPUs. It does not tell a 10-node site *which
layer* is broken. The IPDPS 2026 paper says outright that GPU-cluster variability
has not been analyzed as systematically as CPU HPC. A small-cluster,
layer-by-layer localization protocol with two validated case studies is a real
contribution — and the rail-affinity trap alone is worth documenting, because any
site can hit it and silently publish a wrong number.

## 2.4 Add energy — the cheapest new dimension we have

**What our data shows.** Nothing: we measure no power at all. But gpu-fryer
already runs each precision to a **converged steady state** across all 8 GPUs, so
the hard part (a stable, reproducible load) is done.

**What to run.**
1. Wrap gpu-fryer with DCGM/NVML sampling; report **TFLOP/J by precision**
   (FP32 / BF16 / FP8) across all 80 GPUs. Per-GPU energy spread is as
   interesting as the mean.
2. Same instrumentation on Megatron — **J per token** and J per iteration, 1/2/3
   nodes, where inter-node communication adds energy but no FLOPs.
3. Power-cap sweep (`nvidia-smi -pl`) to find the efficiency knee, as the H100/H200
   power-capping study did for the previous generation.
4. Measure **idle and allocated-but-unused** draw across the 10 nodes — the
   execution-idle paper's concern, and a direct input to scheduler policy.

**Why it matters.** Blackwell cluster-scale energy data is scarce (one 2026
multi-generation study had just 8 B200s). We have 80. And unlike most of this
literature, we can report energy *per precision at a converged state* rather than
averaged over a mixed workload.

## 2.5 The FP64 question — what a B200 is actually worth to simulation users

**What our data shows.** We characterize FP32, BF16, FP8 and a 7B transformer.
We have **no FP64 number**, and no traditional HPC application. For an HPC center
where many users run CFD, MD, or QCD rather than LLMs, that is the gap that
matters most operationally.

**What to run.**
1. **HPL (FP64)** and **HPCG** from the NGC HPC-Benchmarks container, 1/2/4/8
   nodes. Report FP64 TFLOP/s per GPU and the **FP64:BF16:FP8 ratio** on this
   silicon.
2. **HPL-MxP** — mixed-precision HPL — to quantify how much of the FP64 shortfall
   low precision plus iterative refinement can recover in practice.
3. At least two real applications with different bottlenecks, e.g. **GROMACS**
   (latency/strong-scaling bound) and **Quantum ESPRESSO** or **LAMMPS**
   (FP64/communication bound).
4. Frame against H100/H200 where a comparison is available.

**Why it matters.** The vendor's own documentation flags weak native FP64 on the
B300 sibling. A measured, independent answer for B200 — "here is the FP64 rate,
here is what HPL-MxP recovers, here is what it means for your Quantum ESPRESSO
job" — is directly useful to every HPC center making the same procurement
decision, and it is not in the literature.

## 2.6 Push Megatron into the regime where the interconnect actually bites

**What our data shows.** `output-megatron/summary.md` is TP=1, PP=1,
data-parallel only, ~7B model, up to 3 nodes / 24 GPUs. Weak-scaling efficiency
is **97.8-99.1%** — an excellent result, but it also means the fabric is barely
stressed: pure data-parallel gradient all-reduce at this model size hides
everything we found in section 2.1.

**What to run.**
1. **TP>1 across the node boundary** — the configuration that converts our 60%
   all_reduce and 12% alltoall into visible end-to-end loss.
2. **Pipeline parallelism** (PP=2,4) to exercise point-to-point sendrecv, which
   is our *strongest* collective (99% of line rate) — the contrast is the point.
3. **An MoE model with expert parallelism** — the production consumer of
   all-to-all, and the direct end-to-end test of section 2.1.
4. **Scale to all available nodes** (7+ new nodes now exist beyond the original
   three), with strong scaling as well as weak.
5. A **long-sequence** configuration (context parallelism) to shift the
   compute/communication balance again.

**Why it matters.** It closes the loop: a microbenchmark deficiency is only
interesting if it costs real training throughput. Right now our own workload
cannot see it.

## 2.7 Run-to-run and node-to-node variability, properly

**What our data shows.** We have hints, never characterized: `node5602-c1` runs
**1.4% below** the 7-node mean and is worst on root-anchored collectives
(**-4.8%** on reduce_scatter); NCCL 2-node pairs differ by up to **5.0%**; per-node
gpu-fryer means spread **1.7%** across 10 nodes. Every one of these is a *single*
measurement, so we cannot say whether any of it is signal.

**What to run.**
1. **Repeat** the gpu-fryer and NCCL suites N>=10 times per node and report
   distributions, not point values.
2. Build a per-GPU/per-node **fingerprint** and track it over weeks — a
   site-scale, small-N version of what Guard does at 10k scale.
3. Correlate outliers with temperature, clocks, and throttle flags already in the
   gpu-fryer output.
4. Determine an **acceptance threshold**: how far below the fleet median is a
   real fault versus normal spread? We currently guess.

**Why it matters.** The IPDPS 2026 and TPDS 2026 papers both argue this is
under-studied on GPU clusters. A 10-node / 80-GPU longitudinal dataset with
per-precision and per-collective fingerprints is a tractable contribution, and it
feeds directly back into the acceptance test of section 2.3.

## 2.8 Two smaller items worth closing

- **`hypercube` fails on all 10 nodes.** Confirmed as a known nccl-tests issue
  rather than a node fault, but it has never been pinned to an upstream cause or
  version. Worth 30 minutes to identify and cite, so the FAIL in our tables is
  explained rather than merely excused.
- **The OS/kernel natural experiment.** `notes.md` records that node5500/5501
  were reinstalled EL8 -> EL10 (kernel 4.18 -> 6.12) **at the same time** IOMMU
  was disabled, so the 2.7x GPU-read recovery is confounded between two changes.
  The `b200-ubuntu/` tree gives a third OS point. If a node can be booted with
  `iommu=pt` on the *current* kernel, the confound resolves in one job — and
  "IOMMU mode costs 2.7x on the GPU-read RDMA path" is a cleanly attributable,
  broadly useful result.

---

# 3. Suggested priority

If the goal is a publishable contribution rather than more site data, the ranking
is:

1. **2.3 — fault-localization methodology.** Most differentiated, most of the
   work already done, fills a stated gap in the literature.
2. **2.5 — FP64 and real HPC applications.** Largest gap between what is
   published and what an HPC center needs to know.
3. **2.1 + 2.6 — alltoall/gather deficiency, then prove the cost end to end.**
   Strongest data we already hold, and a clean microbenchmark-to-application arc.
4. **2.4 — energy.** Cheapest new dimension; 80 GPUs against a literature with 8.
5. **2.7 — variability.** Highest value long-term, but needs weeks of repeated
   runs before it says anything.

Items 2.2 and 2.8 are low-cost and can be interleaved with any of the above.

---

*Measurement sources in this directory: `out-gpu-fryer/summary.md`,
`out-nccl-1node/summary.md`, `out-nccl-2node/summary.md`,
`out-nccl-2node-sharp/summary.md`, `out-ibwrite/`, `output-megatron/summary.md`,
`notes.md`, `notes-aicr.md`, `sharp.md`, `aicr-handoff/`.*
