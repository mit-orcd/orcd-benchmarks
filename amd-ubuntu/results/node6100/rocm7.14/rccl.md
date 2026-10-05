# RCCL Collective Communications — MI355X x8, XGMI

System: 8 x AMD Instinct MI355X (gfx950), ROCm 7.2.4, XGMI all-to-all (K8 mesh, every pair 1 hop). Built natively for gfx950 with no `HSA_OVERRIDE_GFX_VERSION`, so absolute numbers may exceed the Dell Cloud gfx942-override run.

Source runs: rccl_all_20261002_161803, rccl_tests_20261002_164018

`busbw` is steady-state bytes crossing the wire per unit time, normalized for each algorithm's theoretical data movement -- the comparable metric across N and across collectives. All figures below are busbw at the top message size.

> **Headline: the non-power-of-2 cliff reproduces on this machine.** Every ring-based collective loses 67-81% of its bandwidth at N=5/6/7 versus the power-of-2 arities either side of it. The newer ROCm 7.2.4 / RCCL 2.30.4 stack makes it ~20% shallower than the Dell Cloud baseline but does **not** fix it. Root-cause analysis lives in the Dell Cloud writeups — [`summary-power2.md`](../../dell-cloud/rccl-tests/summary-power2.md) (the investigation), [`summary-rccl.md`](../../dell-cloud/rccl-tests/summary-rccl.md) (measured vs fabric spec), and [`notes-amd.md`](../../dell-cloud/rccl-tests/notes-amd.md) (why it shipped). This report reproduces and quantifies the cliff; it does not re-derive it.

## 1. Measured results

### 1.1 Full collective sweep

| collective | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | cliff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 59.3 | 88.0 | 167.5 | 45.9 | 44.5 | 43.8 | 389.0 | **79% down** |
| all_reduce | 59.6 | 93.6 | 168.8 | 48.1 | 47.6 | 48.0 | 395.8 | **77% down** |
| alltoall | 58.7 | 112.8 | 155.0 | 61.5 | 61.5 | 61.8 | 347.6 | **67% down** |
| alltoallv | 58.7 | 87.8 | 113.9 | 45.4 | 41.8 | 41.1 | 218.1 | **68% down** |
| broadcast | 62.0 | 87.3 | 176.8 | 39.9 | 39.5 | 39.7 | 389.1 | **81% down** |
| gather | 61.8 | 122.6 | 182.2 | 73.2 | 75.2 | 87.4 | 424.8 | **67% down** |
| reduce | 61.5 | 100.6 | 170.6 | 46.4 | 45.8 | 46.1 | 322.0 | **75% down** |
| reduce_scatter | 56.9 | 81.9 | 166.6 | 46.4 | 47.5 | 48.0 | 388.4 | **77% down** |
| scatter | 61.9 | 122.2 | 180.2 | 76.3 | 80.2 | 83.5 | 395.6 | **64% down** |
| sendrecv | 58.5 | 60.8 | 61.0 | 60.5 | 60.3 | 60.4 | 60.3 | none |

![RCCL busbw vs GPU count — the non-power-of-2 cliff](rccl_busbw.png)

Blue = ring-based collectives (need a complete ring across the mesh); orange = pairwise-sendrecv collectives (route directly to/from a root, no ring required). The shaded band is N=5,6,7. Ring-based collectives collapse ~3.5-5x inside it and snap back at N=8; `sendrecv` is flat throughout because a single point-to-point exchange never depends on ring construction. Regenerate with `plot_rccl_busbw.py results/rccl.csv results/rccl_busbw.png`.

### 1.1a Dell Cloud vs amd-ubuntu — full sweep (busbw GB/s at top message size)

Same silicon, same fabric (8 x MI355X, XGMI 4th gen K8 mesh) on both hosts; the only difference is software (ROCm 7.2.3 + gfx942 alias on Dell Cloud vs ROCm 7.2.4 native gfx950 here). Each cell is `Dell / AMD (AMD÷Dell)`.

| Collective | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|
| all_gather | 60.6/59.3 (0.98x) | 71.1/88.0 (1.24x) | 158.7/167.5 (1.06x) | 35.4/45.9 (1.30x) | 34.9/44.5 (1.28x) | 34.9/43.8 (1.26x) | 365.8/389.0 (1.06x) |
| all_reduce | 61.3/59.6 (0.97x) | 75.0/93.6 (1.25x) | 166.5/168.8 (1.01x) | 38.4/48.1 (1.26x) | 38.4/47.6 (1.24x) | 38.2/48.0 (1.26x) | 381.3/395.8 (1.04x) |
| alltoall‡ | 58.4/58.7 (1.00x) | 61.8/**112.8** (1.83x) | 155.2/155.0 (1.00x) | 44.1/61.5 (1.39x) | 45.6/61.5 (1.35x) | 44.3/61.8 (1.40x) | 360.9/347.6 (0.96x) |
| broadcast | 63.5/62.0 (0.98x) | 68.1/87.3 (1.28x) | 169.2/176.8 (1.05x) | 34.1/39.9 (1.17x) | 33.9/39.5 (1.17x) | 33.8/39.7 (1.17x) | 377.3/389.1 (1.03x) |
| gather | 72.1/61.8 (0.86x) | 78.3/**122.6** (1.57x) | 211.6/182.2 (0.86x) | 69.4/73.2 (1.06x) | 68.8/75.2 (1.09x) | 70.3/87.4 (1.24x) | 444.1/424.8 (0.96x) |
| reduce | 72.9/**61.5** (0.84x) | 86.5/100.6 (1.16x) | 197.4/170.6 (0.86x) | 43.6/46.4 (1.06x) | 42.9/45.8 (1.07x) | 43.1/46.1 (1.07x) | 358.5/322.0 (0.90x) |
| reduce_scatter | 60.6/56.9 (0.94x) | 71.0/81.9 (1.15x) | 165.1/166.6 (1.01x) | 39.6/46.4 (1.17x) | 39.6/47.5 (1.20x) | 40.5/48.0 (1.19x) | 407.7/388.4 (0.95x) |
| scatter | 63.1/61.9 (0.98x) | 71.5/**122.2** (1.71x) | 191.6/180.2 (0.94x) | 65.3/76.3 (1.17x) | 65.6/80.2 (1.22x) | 66.4/83.5 (1.26x) | 426.4/395.6 (0.93x) |
| sendrecv | 59.2/58.5 (0.99x) | 60.3/60.8 (1.01x) | 60.6/61.0 (1.01x) | 43.8/60.5 (1.38x) | 43.8/60.3 (1.38x) | 43.4/60.4 (1.39x) | 53.2/60.3 (1.13x) |

‡ measured here at a smaller top message size than Dell Cloud's 8 GiB (both sides cap `alltoall`/`alltoallv` early to survive the N=5 OOM that killed Dell Cloud's alltoallv run — see run-rccl-all.sh `ALLTOALL_MAX`). busbw plateaus well before 8 GiB for every collective measured (Dell Cloud's own finding, summary-rccl.md §1.1), so the smaller cap should still land in the flat region, but it is not a strictly identical measurement and is flagged rather than presented as one.

Ratio ranges from **0.84x** (`reduce` N=2) to **1.83x** (`alltoall` N=3). Bold cells are >1.5x or <0.85x — outside what run-to-run noise on identical hardware would explain.

#### Why is amd-ubuntu faster? Only where the ring breaks.

Averaging the ratio per GPU count separates two very different stories:

| N | mean AMD/Dell | power of 2? |
|---:|---:|---|
| 2 | 0.95x | **yes** |
| 3 | 1.36x | no |
| 4 | 0.98x | **yes** |
| 5 | 1.22x | no |
| 6 | 1.22x | no |
| 7 | 1.25x | no |
| 8 | 1.00x | **yes** |

**Power-of-2 arities (N=2,4,8): 0.97x — parity.** **Non-power-of-2 (N=3,5,6,7): 1.26x — a real gain.**

The interpretation: at power-of-2 N, RCCL builds a clean ring that already saturates the fabric on both hosts, so there is nothing left for a newer stack to win — and indeed amd-ubuntu is fractionally *slower* at N=2/4 (0.95-0.98x), within run-to-run noise. The gain appears exclusively where RCCL falls back to a degraded path. Newer ROCm/RCCL (7.2.4 vs 7.2.3) evidently improved that fallback, not the optimal path.

**The cliff still exists — the newer stack only makes it ~20% shallower.** all_reduce still collapses from 169.5 GB/s at N=4 to 48.0 GB/s at N=5 on this host (a **3.5x drop**); it was 166.5 -> 38.4 on Dell Cloud (4.3x). Every ring-based collective still shows a 67-81% drop in the cliff column of section 1.1. The structural problem — no clean ring at non-power-of-2 arities on a K8 mesh — is unchanged, and is a topology/algorithm limitation rather than something a stack upgrade resolves.

> **Root cause is analysed in the Dell Cloud writeups, not repeated here.** This run reproduces the cliff on a second machine with a newer stack; it does not re-derive why it happens. See:
>
> - [`../../dell-cloud/rccl-tests/summary-power2.md`](../../dell-cloud/rccl-tests/summary-power2.md) — the empirical investigation: 5 configs x 2 collectives x N=2..8, establishing that the cliff lives in the RCCL/xGMI layer rather than in Megatron-LM, and that no algorithm knob recovers it.
> - [`../../dell-cloud/rccl-tests/summary-rccl.md`](../../dell-cloud/rccl-tests/summary-rccl.md) sections 1 and 7 — the measured gap against Infinity Fabric paper spec, and why gather/scatter/sendrecv escape it (pairwise sendrecv needs no closed ring).
> - [`../../dell-cloud/rccl-tests/notes-amd.md`](../../dell-cloud/rccl-tests/notes-amd.md) — why the gap shipped: MSCCL plans only exist for `-8n-` arities, AMD's customers run at whole-node multiples of 8, and the structural fix is UALink in MI400 rather than multi-year MSCCL plan authoring.

Note this also means the *headline* comparison is not 'newer software is uniformly faster'. On the paths that matter most for real workloads (TP=8, DP=8 — both power-of-2) the two stacks are equivalent.

##### Which component is responsible?

**Most likely the RCCL library version — 2.27.7 (Dell) -> 2.30.4 (here) — specifically its ring-construction logic.**

The reasoning is that *the gain is arity-specific*. If the cause were faster kernels (gfx950 codegen, newer compiler, newer driver), the speedup would show up at every N — the same reduction kernels run at N=8 as at N=5. It does not: N=8 is 1.00x. Something that helps only where a clean ring cannot be built is topology/algorithm selection logic, which lives in `librccl`, not in codegen.

Two candidates are ruled out by the config sweep in section 2:

- **MSCCL — not it.** `no_mscll` is identical to `default` at every N, and this host has **no MSCCL plans installed at all** (`/opt/rocm/share/rccl/msccl-algorithms/` is empty). Dell Cloud documented having them, so if anything Dell held the advantage.
- **Algorithm knobs — not it.** `ring` and `proto_simple` also match `default` exactly; `tree` is dramatically *worse* (167 vs 396 GB/s at N=8). None of these knobs explain the delta.

What cannot be separated from this data — genuinely confounded between the two hosts:

| Variable | Dell Cloud | amd-ubuntu |
|---|---|---|
| RCCL | 2.27.7 | **2.30.4** |
| ROCm | 7.2.3 | 7.2.4 |
| RCCL kernels | gfx942 alias (`HSA_OVERRIDE_GFX_VERSION`) | native gfx950 |
| Execution | Singularity container | host-native |
| amdgpu driver | 6.16.13 | 6.19.14 |

RCCL ships *with* ROCm, so "ROCm version" and "RCCL version" moved together and cannot be separated by observation alone.

**The decisive test**, if the attribution matters: run the same rccl-tests binary against a different RCCL while holding hardware and driver fixed — e.g. inside `rocm/primus:v26.5`, which carries its own RCCL. If non-power-of-2 tracks the RCCL version while power-of-2 does not move, that confirms library logic. ~20 min. The practical conclusion is unchanged either way, since production configs are all power-of-2 where the two stacks are equivalent.

### 1.2 Infinity Fabric paper spec vs measured ceilings

Each GPU has 7 xGMI links wired point-to-point to the other 7 GPUs. On-node bandwidth telemetry reports N/A on this driver build, so these are AMD's published MI350-series peaks:

| Quantity | Spec |
|---|---:|
| Per xGMI link, bidirectional | **153.6 GB/s** |
| Per xGMI link, per direction | 76.8 GB/s |
| Per-GPU aggregate (x7 links), bidirectional | **1075.2 GB/s** |
| Per-GPU aggregate (x7 links), per direction | 537.6 GB/s |

The comparable ceiling depends on how many links the *specific* collective engages: sendrecv lights one link per pair, while a ring at N drives min(N, 7) links concurrently. Comparing every row to the full 7-link aggregate would understate small-N results.

| Collective | N | Measured (GB/s) | Ceiling (GB/s) | Basis | Achieved |
|---|---:|---:|---:|---|---:|
| all_gather | 2 | 59.54 | 153.6 | 2-link ring x 1 direction | 39% |
| all_gather | 2 | 59.32 | 153.6 | 2-link ring x 1 direction | 39% |
| all_gather | 3 | 87.36 | 230.4 | 3-link ring x 1 direction | 38% |
| all_gather | 3 | 88.01 | 230.4 | 3-link ring x 1 direction | 38% |
| all_gather | 4 | 167.43 | 307.2 | 4-link ring x 1 direction | 55% |
| all_gather | 4 | 167.49 | 307.2 | 4-link ring x 1 direction | 55% |
| all_gather | 5 | 45.98 | 384.0 | 5-link ring x 1 direction | **12%** |
| all_gather | 5 | 45.90 | 384.0 | 5-link ring x 1 direction | **12%** |
| all_gather | 6 | 44.53 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_gather | 6 | 44.54 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_gather | 7 | 43.72 | 537.6 | 7-link ring x 1 direction | **8%** |
| all_gather | 7 | 43.81 | 537.6 | 7-link ring x 1 direction | **8%** |
| all_gather | 8 | 388.59 | 537.6 | 7-link ring x 1 direction | 72% |
| all_gather | 8 | 388.97 | 537.6 | 7-link ring x 1 direction | 72% |
| all_reduce | 2 | 59.61 | 153.6 | 2-link ring x 1 direction | 39% |
| all_reduce | 2 | 59.56 | 153.6 | 2-link ring x 1 direction | 39% |
| all_reduce | 3 | 93.89 | 230.4 | 3-link ring x 1 direction | 41% |
| all_reduce | 3 | 93.62 | 230.4 | 3-link ring x 1 direction | 41% |
| all_reduce | 4 | 168.77 | 307.2 | 4-link ring x 1 direction | 55% |
| all_reduce | 4 | 168.80 | 307.2 | 4-link ring x 1 direction | 55% |
| all_reduce | 5 | 48.25 | 384.0 | 5-link ring x 1 direction | **13%** |
| all_reduce | 5 | 48.15 | 384.0 | 5-link ring x 1 direction | **13%** |
| all_reduce | 6 | 47.76 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_reduce | 6 | 47.65 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_reduce | 7 | 47.79 | 537.6 | 7-link ring x 1 direction | **9%** |
| all_reduce | 7 | 47.99 | 537.6 | 7-link ring x 1 direction | **9%** |
| all_reduce | 8 | 395.94 | 537.6 | 7-link ring x 1 direction | 74% |
| all_reduce | 8 | 395.84 | 537.6 | 7-link ring x 1 direction | 74% |
| alltoall | 2 | 58.68 | 153.6 | 2 concurrent pairwise links | 38% |
| alltoall | 3 | 112.84 | 230.4 | 3 concurrent pairwise links | 49% |
| alltoall | 4 | 154.98 | 307.2 | 4 concurrent pairwise links | 50% |
| alltoall | 5 | 61.52 | 384.0 | 5 concurrent pairwise links | **16%** |
| alltoall | 6 | 61.51 | 460.8 | 6 concurrent pairwise links | **13%** |
| alltoall | 7 | 61.78 | 537.6 | 7 concurrent pairwise links | **11%** |
| alltoall | 8 | 347.64 | 537.6 | 7 concurrent pairwise links | 65% |
| alltoallv | 2 | 58.72 | 153.6 | 2 concurrent pairwise links | 38% |
| alltoallv | 3 | 87.80 | 230.4 | 3 concurrent pairwise links | 38% |
| alltoallv | 4 | 113.88 | 307.2 | 4 concurrent pairwise links | 37% |
| alltoallv | 5 | 45.40 | 384.0 | 5 concurrent pairwise links | **12%** |
| alltoallv | 6 | 41.76 | 460.8 | 6 concurrent pairwise links | **9%** |
| alltoallv | 7 | 41.12 | 537.6 | 7 concurrent pairwise links | **8%** |
| alltoallv | 8 | 218.12 | 537.6 | 7 concurrent pairwise links | 41% |
| broadcast | 2 | 62.00 | 153.6 | 2-link ring x 1 direction | 40% |
| broadcast | 3 | 87.26 | 230.4 | 3-link ring x 1 direction | 38% |
| broadcast | 4 | 176.85 | 307.2 | 4-link ring x 1 direction | 58% |
| broadcast | 5 | 39.89 | 384.0 | 5-link ring x 1 direction | **10%** |
| broadcast | 6 | 39.54 | 460.8 | 6-link ring x 1 direction | **9%** |
| broadcast | 7 | 39.66 | 537.6 | 7-link ring x 1 direction | **7%** |
| broadcast | 8 | 389.15 | 537.6 | 7-link ring x 1 direction | 72% |
| gather | 2 | 61.81 | 153.6 | 2 concurrent pairwise links | 40% |
| gather | 3 | 122.63 | 230.4 | 3 concurrent pairwise links | 53% |
| gather | 4 | 182.22 | 307.2 | 4 concurrent pairwise links | 59% |
| gather | 5 | 73.20 | 384.0 | 5 concurrent pairwise links | **19%** |
| gather | 6 | 75.21 | 460.8 | 6 concurrent pairwise links | **16%** |
| gather | 7 | 87.37 | 537.6 | 7 concurrent pairwise links | **16%** |
| gather | 8 | 424.81 | 537.6 | 7 concurrent pairwise links | 79% |
| reduce | 2 | 61.50 | 153.6 | 2-link ring x 1 direction | 40% |
| reduce | 3 | 100.63 | 230.4 | 3-link ring x 1 direction | 44% |
| reduce | 4 | 170.56 | 307.2 | 4-link ring x 1 direction | 56% |
| reduce | 5 | 46.35 | 384.0 | 5-link ring x 1 direction | **12%** |
| reduce | 6 | 45.77 | 460.8 | 6-link ring x 1 direction | **10%** |
| reduce | 7 | 46.07 | 537.6 | 7-link ring x 1 direction | **9%** |
| reduce | 8 | 322.01 | 537.6 | 7-link ring x 1 direction | 60% |
| reduce_scatter | 2 | 56.93 | 153.6 | 2-link ring x 1 direction | 37% |
| reduce_scatter | 3 | 81.91 | 230.4 | 3-link ring x 1 direction | 36% |
| reduce_scatter | 4 | 166.58 | 307.2 | 4-link ring x 1 direction | 54% |
| reduce_scatter | 5 | 46.39 | 384.0 | 5-link ring x 1 direction | **12%** |
| reduce_scatter | 6 | 47.50 | 460.8 | 6-link ring x 1 direction | **10%** |
| reduce_scatter | 7 | 47.98 | 537.6 | 7-link ring x 1 direction | **9%** |
| reduce_scatter | 8 | 388.43 | 537.6 | 7-link ring x 1 direction | 72% |
| scatter | 2 | 61.94 | 153.6 | 2 concurrent pairwise links | 40% |
| scatter | 3 | 122.23 | 230.4 | 3 concurrent pairwise links | 53% |
| scatter | 4 | 180.23 | 307.2 | 4 concurrent pairwise links | 59% |
| scatter | 5 | 76.26 | 384.0 | 5 concurrent pairwise links | **20%** |
| scatter | 6 | 80.23 | 460.8 | 6 concurrent pairwise links | **17%** |
| scatter | 7 | 83.51 | 537.6 | 7 concurrent pairwise links | **16%** |
| scatter | 8 | 395.62 | 537.6 | 7 concurrent pairwise links | 74% |
| sendrecv | 2 | 58.47 | 76.8 | 1 link x 1 direction | 76% |
| sendrecv | 3 | 60.79 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 4 | 60.98 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 5 | 60.47 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 6 | 60.32 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 7 | 60.44 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 8 | 60.34 | 76.8 | 1 link x 1 direction | 79% |

Rows below 25% of their ceiling are bolded: at that level the arity is not constructing a usable communication pattern, rather than merely running inefficiently.

### 1.3 Interconnect comparison — Dell Cloud vs amd-ubuntu vs NVIDIA reference

Dell Cloud and amd-ubuntu are the **same fabric on the same silicon** (8 x MI355X, XGMI 4th gen, K8 direct mesh); only the software stack differs. The NVIDIA rows are **published spec only** — no NCCL run exists on either machine in this repo, so quoting someone else's busbw beside ours would not be like-for-like.

| Machine | Fabric | Topology | Per-link (bidir) | Per-GPU aggregate (bidir) | Per-GPU (per direction) | Measured AllReduce N=8 | % of ceiling |
|---|---|---|---|---:|---:|---:|---:|
| Dell Cloud — 8x MI355X | Infinity Fabric (XGMI) 4th gen | direct mesh (K8, 1 hop) | 153.6 GB/s x7 | 1075.2 GB/s | 537.6 GB/s | 381.27 GB/s | 71% |
| **amd-ubuntu (this host)** — 8x MI355X | Infinity Fabric (XGMI) 4th gen | direct mesh (K8, 1 hop) | 153.6 GB/s x7 | 1075.2 GB/s | 537.6 GB/s | **395.84 GB/s** | 74% |
| NVIDIA H100 SXM (ref) — 8x GPU | NVLink 4 + NVSwitch | switched all-to-all | 25 GB/s x 18 links | 900.0 GB/s | 450.0 GB/s | _not measured (spec only)_ | — |
| NVIDIA B200 SXM (ref) — 8x GPU | NVLink 5 + NVSwitch | switched all-to-all | 50 GB/s x 18 links | 1800.0 GB/s | 900.0 GB/s | _not measured (spec only)_ | — |

Reading:

- **B200's NVLink 5 has ~1.67x the per-GPU fabric bandwidth of MI355X's XGMI** (1800 vs 1075 GB/s bidirectional). H100's NVLink 4 is slightly *below* MI355X (900 GB/s) — the dell-cloud readme's "comparable to NVLink 4" characterisation is right for H100 and wrong for B200.
- **The architectural difference matters more than the headline number.** NVIDIA routes through an NVSwitch, so any subset of GPUs gets full switched all-to-all bandwidth. AMD's mesh is direct point-to-point, which is why ring construction — and therefore the collective arity N — determines how much of the fabric is reachable.
- That is the root of the non-power-of-2 cliff: dell-cloud measured ~38 GB/s at N=5/6/7 (~7% of ceiling) for AllReduce, against 381 GB/s at N=8. A switched fabric has no equivalent failure mode, which is why NVIDIA stopped seeing these cliffs after DGX-1/P100 and why AMD's structural fix is UALink in MI400 rather than more MSCCL plans.

### 1.4 Same-silicon comparison vs Dell Cloud

Both hosts are 8 x MI355X. Dell Cloud ran ROCm 7.2.3 with the gfx942 alias; this host runs ROCm 7.2.4 with native gfx950 code objects.

| Collective | N | Dell Cloud | amd-ubuntu | AMD/Dell |
|---|---:|---:|---:|---:|
| all_reduce | 4 | 166.48 | 168.80 | **1.01x** |
| all_reduce | 8 | 381.27 | 395.84 | **1.04x** |
| gather | 8 | 444.15 | 424.81 | **0.96x** |
| reduce_scatter | 8 | 407.69 | 388.43 | **0.95x** |
| scatter | 8 | 426.40 | 395.62 | **0.93x** |
| sendrecv | 2 | 59.21 | 58.47 | **0.99x** |

## 2. Config sweep — which knob recovers a cliff

### `all_gather` — config comparison

| config | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | cliff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 59.3 | 88.0 | 167.5 | 45.9 | 44.5 | 43.8 | 389.0 | **79% down** |
| no_mscll | 59.5 | 87.7 | 167.4 | 46.0 | 44.5 | 44.0 | 389.7 | **79% down** |
| proto_simple | 59.4 | 91.3 | 167.4 | 45.9 | 44.5 | 43.6 | 390.1 | **79% down** |
| ring | 59.7 | 87.5 | 167.3 | 46.0 | 44.4 | 43.7 | 388.8 | **79% down** |
| tree | 59.5 | 87.1 | 167.5 | 46.0 | 44.5 | 44.0 | 389.3 | **79% down** |

### `all_reduce` — config comparison

| config | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | cliff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 59.6 | 93.6 | 168.8 | 48.1 | 47.6 | 48.0 | 395.8 | **77% down** |
| no_mscll | 59.7 | 93.7 | 168.9 | 48.3 | 47.7 | 47.9 | 396.9 | **77% down** |
| proto_simple | 59.5 | 93.6 | 169.0 | 48.3 | 47.8 | 48.0 | 397.2 | **77% down** |
| ring | 59.6 | 93.6 | 168.9 | 48.2 | 47.8 | 47.8 | 396.3 | **77% down** |
| tree | 27.5 | 25.1 | 62.5 | 15.0 | 15.5 | 16.1 | 170.8 | **83% down** |

## 3. How to read the cliff column

`X% down` means the worst non-power-of-2 N is that much below the mean of the power-of-2 Ns.

- If the `default` config cliffs but `tree`/`ring`/`no_mscll` do not, the recovering knob is a one-env-var workaround and should be adopted.
- If nothing recovers it, the gap is missing RCCL tuning for those arities on gfx950 — RCCL has failed to construct a valid ring, and the fix is upstream.
- Either way the attribution to the *algorithm layer* rests on Part A's RVS `pbqt` (peer-to-peer XGMI) and `pebb` (PCIe) runs coming back clean. Without that, a cliff could equally be a bad link.

This matters for training: a Ring AllReduce at N=8 is the realistic upper bound for data-parallel gradient sync on this box — no Megatron dist-opt tuning can exceed it.

## 4. Reference

### 4.1 RCCL algorithm selection

| Collective | Ring | Tree | PAT | MSCCL | Pairwise sendrecv |
|---|---|---|---|---|---|
| AllReduce | Default large | Default small (degraded on mesh) | — | If plan exists | — |
| AllGather | Default | Falls back to Ring | Available | If plan exists | — |
| ReduceScatter | Default | Falls back to Ring | Available | If plan exists | — |
| Broadcast | Default large | Default small | — | — | — |
| Reduce | Default large | Default small | — | — | — |
| Gather / Scatter | — | — | — | — | Default |
| AllToAll | — | — | — | If plan exists | Default |
| SendRecv | — | — | — | — | Direct |

- **NVLS** (in-network reduction) is not applicable on AMD until UALink ships. **PAT** is a switched-fabric path, not active on an xGMI mesh.
- Forcing `NCCL_ALGO=Tree` on AllGather / ReduceScatter is silently equivalent to Ring.
- MSCCL plans ship in `/opt/rocm/share/rccl/msccl-algorithms/`; everything without a plan falls through to Ring or pairwise sendrecv.

### 4.2 Configuration knobs

| Variable | Value used here | Effect |
|---|---|---|
| `NCCL_ALGO` | `Ring,Tree` | Algorithm pool. `Tree` only affects AllReduce / Broadcast / Reduce. |
| `NCCL_PROTO` | `Simple,LL,LL128` | Wire protocol; `Simple` is effectively the default at large message size. |
| `RCCL_MSCCL_ENABLE` | `1` | Toggle MSCCL plan dispatch. |
| `NCCL_P2P_DISABLE` | `0` | `1` forces host-SHM staging; debug only. |
| `NCCL_SHM_DISABLE` | `0` | Leave on. |
| `NCCL_IB_DISABLE` | `1` | Single-node run, no IB. |
| `NCCL_SOCKET_IFNAME` | `lo` | Bootstrap over loopback. |
| `NCCL_DEBUG` | `WARN` | `INFO` prints algorithm + channel count per call. |
| `HSA_OVERRIDE_GFX_VERSION` | **unset** | Deliberately native gfx950; the gfx942 alias would undercount. |

### 4.3 Process model caveat

rccl-tests runs **one process driving N GPUs** (`-g N`, `MPI=0`), while real training and Primus' `benchmark rccl` run **N processes with 1 GPU each**. These take different code paths inside RCCL. If Part C's collective numbers disagree with these, the process model is the first suspect.

## 5. Source data

| What | Where |
|---|---|
| Raw rccl-tests stdout | `logs/rccl/rccl_*/<coll>_n<N>.log` |
| Config sweep logs | `logs/rccl/rccl_tests_*/<coll>_<cfg>_n<N>.log` |
| Per-run summary | `logs/rccl/rccl_*/rccl_summary.txt` |
| This table as CSV | `results/rccl.csv` |
| Figure | `results/rccl_busbw.png` |

