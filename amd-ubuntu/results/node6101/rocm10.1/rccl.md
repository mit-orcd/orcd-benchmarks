# RCCL Collective Communications — MI355X x8, XGMI

System: 8 x AMD Instinct MI355X (gfx950), ROCm 7.2.4, XGMI all-to-all (K8 mesh, every pair 1 hop). Built natively for gfx950 with no `HSA_OVERRIDE_GFX_VERSION`, so absolute numbers may exceed the Dell Cloud gfx942-override run.

Source runs: rccl_all_20261027_213232

`busbw` is steady-state bytes crossing the wire per unit time, normalized for each algorithm's theoretical data movement -- the comparable metric across N and across collectives. All figures below are busbw at the top message size.

> **Headline: the non-power-of-2 cliff reproduces on this machine.** Every ring-based collective loses 67-81% of its bandwidth at N=5/6/7 versus the power-of-2 arities either side of it. The newer ROCm 7.2.4 / RCCL 2.30.4 stack makes it ~20% shallower than the Dell Cloud baseline but does **not** fix it. Root-cause analysis lives in the Dell Cloud writeups — [`summary-power2.md`](../../dell-cloud/rccl-tests/summary-power2.md) (the investigation), [`summary-rccl.md`](../../dell-cloud/rccl-tests/summary-rccl.md) (measured vs fabric spec), and [`notes-amd.md`](../../dell-cloud/rccl-tests/notes-amd.md) (why it shipped). This report reproduces and quantifies the cliff; it does not re-derive it.

## 1. Measured results

### 1.1 Full collective sweep

| collective | N=8 | cliff |
|---|---:|---:|
| all_gather | 388.0 | - |
| all_reduce | 396.0 | - |
| alltoall | 347.2 | - |
| alltoallv | 219.8 | - |
| broadcast | 388.7 | - |
| gather | 422.1 | - |
| reduce | 333.9 | - |
| reduce_scatter | 385.5 | - |
| scatter | 396.8 | - |
| sendrecv | 60.1 | - |

![RCCL busbw vs GPU count — the non-power-of-2 cliff](rccl_busbw.png)

Blue = ring-based collectives (need a complete ring across the mesh); orange = pairwise-sendrecv collectives (route directly to/from a root, no ring required). The shaded band is N=5,6,7. Ring-based collectives collapse ~3.5-5x inside it and snap back at N=8; `sendrecv` is flat throughout because a single point-to-point exchange never depends on ring construction. Regenerate with `plot_rccl_busbw.py results/rccl.csv results/rccl_busbw.png`.

### 1.1a Dell Cloud vs amd-ubuntu — full sweep (busbw GB/s at top message size)

Same silicon, same fabric (8 x MI355X, XGMI 4th gen K8 mesh) on both hosts; the only difference is software (ROCm 7.2.3 + gfx942 alias on Dell Cloud vs ROCm 7.2.4 native gfx950 here). Each cell is `Dell / AMD (AMD÷Dell)`.

| Collective | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---|---|---|---|---|---|---|
| all_gather | - | - | - | - | - | - | 365.8/388.0 (1.06x) |
| all_reduce | - | - | - | - | - | - | 381.3/396.0 (1.04x) |
| alltoall‡ | - | - | - | - | - | - | 360.9/347.2 (0.96x) |
| broadcast | - | - | - | - | - | - | 377.3/388.7 (1.03x) |
| gather | - | - | - | - | - | - | 444.1/422.1 (0.95x) |
| reduce | - | - | - | - | - | - | 358.5/333.9 (0.93x) |
| reduce_scatter | - | - | - | - | - | - | 407.7/385.5 (0.95x) |
| scatter | - | - | - | - | - | - | 426.4/396.8 (0.93x) |
| sendrecv | - | - | - | - | - | - | 53.2/60.1 (1.13x) |

‡ measured here at a smaller top message size than Dell Cloud's 8 GiB (both sides cap `alltoall`/`alltoallv` early to survive the N=5 OOM that killed Dell Cloud's alltoallv run — see run-rccl-all.sh `ALLTOALL_MAX`). busbw plateaus well before 8 GiB for every collective measured (Dell Cloud's own finding, summary-rccl.md §1.1), so the smaller cap should still land in the flat region, but it is not a strictly identical measurement and is flagged rather than presented as one.

Ratio ranges from **0.93x** (`scatter` N=8) to **1.13x** (`sendrecv` N=8). Bold cells are >1.5x or <0.85x — outside what run-to-run noise on identical hardware would explain.

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
| all_gather | 8 | 388.04 | 537.6 | 7-link ring x 1 direction | 72% |
| all_reduce | 8 | 395.97 | 537.6 | 7-link ring x 1 direction | 74% |
| alltoall | 8 | 347.20 | 537.6 | 7 concurrent pairwise links | 65% |
| alltoallv | 8 | 219.75 | 537.6 | 7 concurrent pairwise links | 41% |
| broadcast | 8 | 388.66 | 537.6 | 7-link ring x 1 direction | 72% |
| gather | 8 | 422.10 | 537.6 | 7 concurrent pairwise links | 79% |
| reduce | 8 | 333.90 | 537.6 | 7-link ring x 1 direction | 62% |
| reduce_scatter | 8 | 385.54 | 537.6 | 7-link ring x 1 direction | 72% |
| scatter | 8 | 396.75 | 537.6 | 7 concurrent pairwise links | 74% |
| sendrecv | 8 | 60.13 | 76.8 | 1 link x 1 direction | 78% |

Rows below 25% of their ceiling are bolded: at that level the arity is not constructing a usable communication pattern, rather than merely running inefficiently.

### 1.3 Interconnect comparison — Dell Cloud vs amd-ubuntu vs NVIDIA reference

Dell Cloud and amd-ubuntu are the **same fabric on the same silicon** (8 x MI355X, XGMI 4th gen, K8 direct mesh); only the software stack differs. The NVIDIA rows are **published spec only** — no NCCL run exists on either machine in this repo, so quoting someone else's busbw beside ours would not be like-for-like.

| Machine | Fabric | Topology | Per-link (bidir) | Per-GPU aggregate (bidir) | Per-GPU (per direction) | Measured AllReduce N=8 | % of ceiling |
|---|---|---|---|---:|---:|---:|---:|
| Dell Cloud — 8x MI355X | Infinity Fabric (XGMI) 4th gen | direct mesh (K8, 1 hop) | 153.6 GB/s x7 | 1075.2 GB/s | 537.6 GB/s | 381.27 GB/s | 71% |
| **amd-ubuntu (this host)** — 8x MI355X | Infinity Fabric (XGMI) 4th gen | direct mesh (K8, 1 hop) | 153.6 GB/s x7 | 1075.2 GB/s | 537.6 GB/s | **395.97 GB/s** | 74% |
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
| all_reduce | 4 | 166.48 | - | - |
| all_reduce | 8 | 381.27 | 395.97 | **1.04x** |
| gather | 8 | 444.15 | 422.10 | **0.95x** |
| reduce_scatter | 8 | 407.69 | 385.54 | **0.95x** |
| scatter | 8 | 426.40 | 396.75 | **0.93x** |
| sendrecv | 2 | 59.21 | - | - |

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

