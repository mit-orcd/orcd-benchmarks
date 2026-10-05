# RCCL Collective Communications — MI355X x8, XGMI

System: 8 x AMD Instinct MI355X (gfx950), ROCm 7.2.4, XGMI all-to-all (K8 mesh, every pair 1 hop). Built natively for gfx950 with no `HSA_OVERRIDE_GFX_VERSION`, so absolute numbers may exceed the Dell Cloud gfx942-override run.

Source runs: rccl_all_20261002_103010, rccl_tests_20261002_105207

`busbw` is steady-state bytes crossing the wire per unit time, normalized for each algorithm's theoretical data movement -- the comparable metric across N and across collectives. All figures below are busbw at the top message size.

> **Headline: the non-power-of-2 cliff reproduces on this machine.** Every ring-based collective loses 67-81% of its bandwidth at N=5/6/7 versus the power-of-2 arities either side of it. The newer ROCm 7.2.4 / RCCL 2.30.4 stack makes it ~20% shallower than the Dell Cloud baseline but does **not** fix it. Root-cause analysis lives in the Dell Cloud writeups — [`summary-power2.md`](../../dell-cloud/rccl-tests/summary-power2.md) (the investigation), [`summary-rccl.md`](../../dell-cloud/rccl-tests/summary-rccl.md) (measured vs fabric spec), and [`notes-amd.md`](../../dell-cloud/rccl-tests/notes-amd.md) (why it shipped). This report reproduces and quantifies the cliff; it does not re-derive it.

## 1. Measured results

### 1.1 Full collective sweep

| collective | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | cliff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 59.5 | 91.4 | 167.8 | 46.0 | 44.6 | 43.6 | 389.7 | **79% down** |
| all_reduce | 59.9 | 93.4 | 169.1 | 48.2 | 47.6 | 47.6 | 398.0 | **77% down** |
| alltoall | 59.1 | 113.0 | 154.8 | 61.2 | 61.5 | 61.8 | 347.5 | **67% down** |
| alltoallv | 59.1 | 87.8 | 113.9 | 45.2 | 41.6 | 40.8 | 220.4 | **69% down** |
| broadcast | 62.7 | 87.4 | 176.4 | 40.0 | 39.1 | 39.7 | 389.7 | **81% down** |
| gather | 61.8 | 122.7 | 182.3 | 70.7 | 76.2 | 83.8 | 422.8 | **68% down** |
| reduce | 61.8 | 100.7 | 170.0 | 45.8 | 45.6 | 45.6 | 328.6 | **76% down** |
| reduce_scatter | 57.6 | 81.9 | 166.3 | 46.7 | 47.6 | 47.8 | 388.9 | **77% down** |
| scatter | 62.1 | 122.3 | 180.3 | 75.7 | 79.9 | 84.2 | 396.1 | **64% down** |
| sendrecv | 58.4 | 60.5 | 60.9 | 60.1 | 60.6 | 60.4 | 60.5 | none |

![RCCL busbw vs GPU count — the non-power-of-2 cliff](rccl_busbw.png)

Blue = ring-based collectives (need a complete ring across the mesh); orange = pairwise-sendrecv collectives (route directly to/from a root, no ring required). The shaded band is N=5,6,7. Ring-based collectives collapse ~3.5-5x inside it and snap back at N=8; `sendrecv` is flat throughout because a single point-to-point exchange never depends on ring construction. Regenerate with `plot_rccl_busbw.py results/rccl.csv results/rccl_busbw.png`.

### 1.1a Dell Cloud vs amd-ubuntu — full sweep (busbw GB/s at top message size)

Same silicon, same fabric (8 x MI355X, XGMI 4th gen K8 mesh) on both hosts; the only difference is software (ROCm 7.2.3 + gfx942 alias on Dell Cloud vs ROCm 7.2.4 native gfx950 here). Each cell is `Dell / AMD (AMD÷Dell)`.

| Collective | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|
| all_gather | 60.6/59.5 (0.98x) | 71.1/91.4 (1.29x) | 158.7/167.8 (1.06x) | 35.4/46.0 (1.30x) | 34.9/44.6 (1.28x) | 34.9/43.6 (1.25x) | 365.8/389.7 (1.07x) |
| all_reduce | 61.3/59.9 (0.98x) | 75.0/93.4 (1.25x) | 166.5/169.1 (1.02x) | 38.4/48.2 (1.26x) | 38.4/47.6 (1.24x) | 38.2/47.6 (1.25x) | 381.3/398.0 (1.04x) |
| alltoall‡ | 58.4/59.1 (1.01x) | 61.8/**113.0** (1.83x) | 155.2/154.8 (1.00x) | 44.1/61.2 (1.39x) | 45.6/61.5 (1.35x) | 44.3/61.8 (1.40x) | 360.9/347.5 (0.96x) |
| broadcast | 63.5/62.7 (0.99x) | 68.1/87.4 (1.28x) | 169.2/176.4 (1.04x) | 34.1/40.0 (1.17x) | 33.9/39.1 (1.15x) | 33.8/39.7 (1.17x) | 377.3/389.7 (1.03x) |
| gather | 72.1/61.8 (0.86x) | 78.3/**122.7** (1.57x) | 211.6/182.3 (0.86x) | 69.4/70.7 (1.02x) | 68.8/76.2 (1.11x) | 70.3/83.8 (1.19x) | 444.1/422.8 (0.95x) |
| reduce | 72.9/**61.8** (0.85x) | 86.5/100.7 (1.16x) | 197.4/170.0 (0.86x) | 43.6/45.8 (1.05x) | 42.9/45.6 (1.06x) | 43.1/45.6 (1.06x) | 358.5/328.6 (0.92x) |
| reduce_scatter | 60.6/57.6 (0.95x) | 71.0/81.9 (1.15x) | 165.1/166.3 (1.01x) | 39.6/46.7 (1.18x) | 39.6/47.6 (1.20x) | 40.5/47.8 (1.18x) | 407.7/388.9 (0.95x) |
| scatter | 63.1/62.1 (0.98x) | 71.5/**122.3** (1.71x) | 191.6/180.3 (0.94x) | 65.3/75.7 (1.16x) | 65.6/79.9 (1.22x) | 66.4/84.2 (1.27x) | 426.4/396.1 (0.93x) |
| sendrecv | 59.2/58.4 (0.99x) | 60.3/60.5 (1.00x) | 60.6/60.9 (1.00x) | 43.8/60.1 (1.37x) | 43.8/60.6 (1.38x) | 43.4/60.4 (1.39x) | 53.2/60.5 (1.14x) |

‡ measured here at a smaller top message size than Dell Cloud's 8 GiB (both sides cap `alltoall`/`alltoallv` early to survive the N=5 OOM that killed Dell Cloud's alltoallv run — see run-rccl-all.sh `ALLTOALL_MAX`). busbw plateaus well before 8 GiB for every collective measured (Dell Cloud's own finding, summary-rccl.md §1.1), so the smaller cap should still land in the flat region, but it is not a strictly identical measurement and is flagged rather than presented as one.

Ratio ranges from **0.85x** (`reduce` N=2) to **1.83x** (`alltoall` N=3). Bold cells are >1.5x or <0.85x — outside what run-to-run noise on identical hardware would explain.

#### Why is amd-ubuntu faster? Only where the ring breaks.

Averaging the ratio per GPU count separates two very different stories:

| N | mean AMD/Dell | power of 2? |
|---:|---:|---|
| 2 | 0.95x | **yes** |
| 3 | 1.36x | no |
| 4 | 0.98x | **yes** |
| 5 | 1.21x | no |
| 6 | 1.22x | no |
| 7 | 1.24x | no |
| 8 | 1.00x | **yes** |

**Power-of-2 arities (N=2,4,8): 0.98x — parity.** **Non-power-of-2 (N=3,5,6,7): 1.26x — a real gain.**

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
| all_gather | 2 | 59.85 | 153.6 | 2-link ring x 1 direction | 39% |
| all_gather | 2 | 59.47 | 153.6 | 2-link ring x 1 direction | 39% |
| all_gather | 3 | 88.19 | 230.4 | 3-link ring x 1 direction | 38% |
| all_gather | 3 | 91.38 | 230.4 | 3-link ring x 1 direction | 40% |
| all_gather | 4 | 167.49 | 307.2 | 4-link ring x 1 direction | 55% |
| all_gather | 4 | 167.77 | 307.2 | 4-link ring x 1 direction | 55% |
| all_gather | 5 | 46.05 | 384.0 | 5-link ring x 1 direction | **12%** |
| all_gather | 5 | 46.05 | 384.0 | 5-link ring x 1 direction | **12%** |
| all_gather | 6 | 44.54 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_gather | 6 | 44.59 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_gather | 7 | 43.80 | 537.6 | 7-link ring x 1 direction | **8%** |
| all_gather | 7 | 43.62 | 537.6 | 7-link ring x 1 direction | **8%** |
| all_gather | 8 | 389.67 | 537.6 | 7-link ring x 1 direction | 72% |
| all_gather | 8 | 389.68 | 537.6 | 7-link ring x 1 direction | 72% |
| all_reduce | 2 | 59.33 | 153.6 | 2-link ring x 1 direction | 39% |
| all_reduce | 2 | 59.92 | 153.6 | 2-link ring x 1 direction | 39% |
| all_reduce | 3 | 93.87 | 230.4 | 3-link ring x 1 direction | 41% |
| all_reduce | 3 | 93.41 | 230.4 | 3-link ring x 1 direction | 41% |
| all_reduce | 4 | 168.66 | 307.2 | 4-link ring x 1 direction | 55% |
| all_reduce | 4 | 169.12 | 307.2 | 4-link ring x 1 direction | 55% |
| all_reduce | 5 | 48.42 | 384.0 | 5-link ring x 1 direction | **13%** |
| all_reduce | 5 | 48.20 | 384.0 | 5-link ring x 1 direction | **13%** |
| all_reduce | 6 | 47.64 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_reduce | 6 | 47.64 | 460.8 | 6-link ring x 1 direction | **10%** |
| all_reduce | 7 | 47.55 | 537.6 | 7-link ring x 1 direction | **9%** |
| all_reduce | 7 | 47.61 | 537.6 | 7-link ring x 1 direction | **9%** |
| all_reduce | 8 | 396.69 | 537.6 | 7-link ring x 1 direction | 74% |
| all_reduce | 8 | 398.02 | 537.6 | 7-link ring x 1 direction | 74% |
| alltoall | 2 | 59.14 | 153.6 | 2 concurrent pairwise links | 39% |
| alltoall | 3 | 112.95 | 230.4 | 3 concurrent pairwise links | 49% |
| alltoall | 4 | 154.81 | 307.2 | 4 concurrent pairwise links | 50% |
| alltoall | 5 | 61.25 | 384.0 | 5 concurrent pairwise links | **16%** |
| alltoall | 6 | 61.48 | 460.8 | 6 concurrent pairwise links | **13%** |
| alltoall | 7 | 61.82 | 537.6 | 7 concurrent pairwise links | **11%** |
| alltoall | 8 | 347.51 | 537.6 | 7 concurrent pairwise links | 65% |
| alltoallv | 2 | 59.06 | 153.6 | 2 concurrent pairwise links | 38% |
| alltoallv | 3 | 87.84 | 230.4 | 3 concurrent pairwise links | 38% |
| alltoallv | 4 | 113.87 | 307.2 | 4 concurrent pairwise links | 37% |
| alltoallv | 5 | 45.17 | 384.0 | 5 concurrent pairwise links | **12%** |
| alltoallv | 6 | 41.60 | 460.8 | 6 concurrent pairwise links | **9%** |
| alltoallv | 7 | 40.78 | 537.6 | 7 concurrent pairwise links | **8%** |
| alltoallv | 8 | 220.36 | 537.6 | 7 concurrent pairwise links | 41% |
| broadcast | 2 | 62.67 | 153.6 | 2-link ring x 1 direction | 41% |
| broadcast | 3 | 87.41 | 230.4 | 3-link ring x 1 direction | 38% |
| broadcast | 4 | 176.43 | 307.2 | 4-link ring x 1 direction | 57% |
| broadcast | 5 | 40.00 | 384.0 | 5-link ring x 1 direction | **10%** |
| broadcast | 6 | 39.09 | 460.8 | 6-link ring x 1 direction | **8%** |
| broadcast | 7 | 39.71 | 537.6 | 7-link ring x 1 direction | **7%** |
| broadcast | 8 | 389.74 | 537.6 | 7-link ring x 1 direction | 72% |
| gather | 2 | 61.81 | 153.6 | 2 concurrent pairwise links | 40% |
| gather | 3 | 122.71 | 230.4 | 3 concurrent pairwise links | 53% |
| gather | 4 | 182.28 | 307.2 | 4 concurrent pairwise links | 59% |
| gather | 5 | 70.73 | 384.0 | 5 concurrent pairwise links | **18%** |
| gather | 6 | 76.16 | 460.8 | 6 concurrent pairwise links | **17%** |
| gather | 7 | 83.78 | 537.6 | 7 concurrent pairwise links | **16%** |
| gather | 8 | 422.82 | 537.6 | 7 concurrent pairwise links | 79% |
| reduce | 2 | 61.78 | 153.6 | 2-link ring x 1 direction | 40% |
| reduce | 3 | 100.65 | 230.4 | 3-link ring x 1 direction | 44% |
| reduce | 4 | 170.02 | 307.2 | 4-link ring x 1 direction | 55% |
| reduce | 5 | 45.84 | 384.0 | 5-link ring x 1 direction | **12%** |
| reduce | 6 | 45.59 | 460.8 | 6-link ring x 1 direction | **10%** |
| reduce | 7 | 45.56 | 537.6 | 7-link ring x 1 direction | **8%** |
| reduce | 8 | 328.55 | 537.6 | 7-link ring x 1 direction | 61% |
| reduce_scatter | 2 | 57.60 | 153.6 | 2-link ring x 1 direction | 38% |
| reduce_scatter | 3 | 81.91 | 230.4 | 3-link ring x 1 direction | 36% |
| reduce_scatter | 4 | 166.27 | 307.2 | 4-link ring x 1 direction | 54% |
| reduce_scatter | 5 | 46.71 | 384.0 | 5-link ring x 1 direction | **12%** |
| reduce_scatter | 6 | 47.56 | 460.8 | 6-link ring x 1 direction | **10%** |
| reduce_scatter | 7 | 47.78 | 537.6 | 7-link ring x 1 direction | **9%** |
| reduce_scatter | 8 | 388.87 | 537.6 | 7-link ring x 1 direction | 72% |
| scatter | 2 | 62.13 | 153.6 | 2 concurrent pairwise links | 40% |
| scatter | 3 | 122.31 | 230.4 | 3 concurrent pairwise links | 53% |
| scatter | 4 | 180.30 | 307.2 | 4 concurrent pairwise links | 59% |
| scatter | 5 | 75.73 | 384.0 | 5 concurrent pairwise links | **20%** |
| scatter | 6 | 79.89 | 460.8 | 6 concurrent pairwise links | **17%** |
| scatter | 7 | 84.22 | 537.6 | 7 concurrent pairwise links | **16%** |
| scatter | 8 | 396.11 | 537.6 | 7 concurrent pairwise links | 74% |
| sendrecv | 2 | 58.43 | 76.8 | 1 link x 1 direction | 76% |
| sendrecv | 3 | 60.52 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 4 | 60.86 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 5 | 60.14 | 76.8 | 1 link x 1 direction | 78% |
| sendrecv | 6 | 60.59 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 7 | 60.40 | 76.8 | 1 link x 1 direction | 79% |
| sendrecv | 8 | 60.51 | 76.8 | 1 link x 1 direction | 79% |

Rows below 25% of their ceiling are bolded: at that level the arity is not constructing a usable communication pattern, rather than merely running inefficiently.

### 1.3 Interconnect comparison — Dell Cloud vs amd-ubuntu vs NVIDIA reference

Dell Cloud and amd-ubuntu are the **same fabric on the same silicon** (8 x MI355X, XGMI 4th gen, K8 direct mesh); only the software stack differs. The NVIDIA rows are **published spec only** — no NCCL run exists on either machine in this repo, so quoting someone else's busbw beside ours would not be like-for-like.

| Machine | Fabric | Topology | Per-link (bidir) | Per-GPU aggregate (bidir) | Per-GPU (per direction) | Measured AllReduce N=8 | % of ceiling |
|---|---|---|---|---:|---:|---:|---:|
| Dell Cloud — 8x MI355X | Infinity Fabric (XGMI) 4th gen | direct mesh (K8, 1 hop) | 153.6 GB/s x7 | 1075.2 GB/s | 537.6 GB/s | 381.27 GB/s | 71% |
| **amd-ubuntu (this host)** — 8x MI355X | Infinity Fabric (XGMI) 4th gen | direct mesh (K8, 1 hop) | 153.6 GB/s x7 | 1075.2 GB/s | 537.6 GB/s | **398.02 GB/s** | 74% |
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
| all_reduce | 4 | 166.48 | 169.12 | **1.02x** |
| all_reduce | 8 | 381.27 | 398.02 | **1.04x** |
| gather | 8 | 444.15 | 422.82 | **0.95x** |
| reduce_scatter | 8 | 407.69 | 388.87 | **0.95x** |
| scatter | 8 | 426.40 | 396.11 | **0.93x** |
| sendrecv | 2 | 59.21 | 58.43 | **0.99x** |

## 2. Config sweep — which knob recovers a cliff

### `all_gather` — config comparison

| config | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | cliff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 59.5 | 91.4 | 167.8 | 46.0 | 44.6 | 43.6 | 389.7 | **79% down** |
| no_mscll | 59.4 | 88.5 | 167.3 | 46.2 | 44.5 | 43.5 | 390.6 | **79% down** |
| proto_simple | 59.5 | 88.2 | 167.4 | 45.7 | 44.6 | 43.8 | 389.2 | **79% down** |
| ring | 59.9 | 91.4 | 166.8 | 46.3 | 44.4 | 43.6 | 390.3 | **79% down** |
| tree | 59.7 | 88.0 | 167.4 | 46.2 | 44.4 | 43.6 | 387.9 | **79% down** |

### `all_reduce` — config comparison

| config | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | cliff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 59.9 | 93.4 | 169.1 | 48.2 | 47.6 | 47.6 | 398.0 | **77% down** |
| no_mscll | 59.9 | 93.7 | 168.7 | 48.0 | 47.5 | 47.6 | 396.0 | **77% down** |
| proto_simple | 59.7 | 93.7 | 168.8 | 48.0 | 47.7 | 47.6 | 397.4 | **77% down** |
| ring | 59.8 | 93.7 | 168.9 | 48.5 | 47.5 | 47.5 | 394.4 | **77% down** |
| tree | 28.2 | 26.0 | 62.4 | 15.0 | 15.4 | 15.9 | 172.4 | **83% down** |

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

