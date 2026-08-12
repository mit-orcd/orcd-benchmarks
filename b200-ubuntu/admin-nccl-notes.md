# NCCL notes for the admins — what to test, in order

Action list drawn from the two NCCL summaries on the Ubuntu B200 nodes. **The evidence is in those files; this one holds only what to do about it.**

- `out-nccl-1node/summary.md` — 8 GPUs, one node, NVLink 5 / NVSwitch
- `out-nccl-2node/summary.md` — 16 GPUs, two nodes, InfiniBand + GPUDirect RDMA
- Rocky 8 counterparts: `../b200-nodes/out-nccl-1node/summary.md`, `../b200-nodes/out-nccl-2node/summary.md`, `../b200-nodes/notes.md`

**Nothing below needs doing on node5700 / node5701.** They perform to specification and now serve as a known-good reference to diff against. The one Ubuntu-side item is in "Loose ends" at the end.

> **Node availability blocks most of this right now.** node5502 has been `DOWN+NOT_RESPONDING` since 2026-08-10 and node5500 is held by another user's reservation until 2026-08-14, leaving **one** free Rocky node. Every 2-node step below needs two. Step 2 is the only one that runs on a single node.

---

## What there is to explain

One thing separates Rocky 8 from Ubuntu, and only across the network. It is named here exactly as the summary section it comes from, so it can be looked up:

| What | Size | Read the evidence in |
|------|------|----------------------|
| **Small messages** — time at 1 MiB, per-operation cost | **1.9-3.0x** | `out-nccl-2node/summary.md` § 2.1, *"Small messages"* — and § 2.2, *"The shape of the gap constrains the explanation"* for the candidates. |

Bulk transfer is **not** a difference: NCCL `sendrecv` at 16 GPUs converges to 48.4 GB/s per pair on Rocky 8 against 48.8 on Ubuntu, at 98% of one rail on both, and `ib_write_bw` reaches NDR line rate reading from GPU memory on both clusters (`../b200-nodes/notes.md`).

**Which collectives to watch, and the 10% threshold.** At 16 GiB only three differ by more than 10%, and they are exactly the three not at the fabric ceiling — the other six are saturating the rails, where no software change can help:

| Collective | Difference at 16 GiB | % of HW max | Faster | What it is |
|------------|---------------------:|------------:|--------|------------|
| `all_reduce` | **15.1%** | 67% | Ubuntu | per-operation cost surviving to the largest size (two ring passes) |
| `alltoall` | **11.1%** | 14% | Ubuntu | per-operation cost, worst case (N^2 transfers, 1.1 of 8 rails) |
| `scatter` | **11.1%** | 73% | **Rocky 8** | root-node outbound aggregate — a different mechanism, unresolved |

`all_reduce` and `alltoall` are **not** separate problems from the small-message gap: they are the same per-transfer cost, visible at 16 GiB because neither collective is bandwidth-bound. Fixing the small-message gap should move both. `scatter` is the one place Rocky 8 leads and needs its own explanation.

At 1 MiB, where nothing is fabric-limited, seven of the nine collectives exceed 10%; only `sendrecv` (1.08x) and `gather` (1.09x) stay under, the two NCCL does not split across channels. One reversal to be aware of when reading sweeps: `broadcast` at 16 MiB is 0.84x — Rocky 8 faster by 16% — at that size only, unexplained.

**Use `all_reduce` and `alltoall` as the metrics for every test below.** They carry the largest signal, they are the two that matter for training, and both should move together if a change addresses the real cause. `sendrecv` and `gather` are the controls: they should stay flat whatever is changed.

**Within a single node the difference does not appear either.** `out-nccl-1node/summary.md` § 2.1 puts every collective in the same 73-93% band of the NVLink ceiling on both clusters, with large messages within 2.8%; its § 2.2 spells out what that rules out. Since the single-node path exercises the GPUs, driver, CUDA and NCCL host setup but *not* the verbs stack or the proxy thread, that tie is what localises the small-message cost to the network stack rather than the launch path (1.10x within a node against 2.23x across two).

## Already ruled out — do not spend time on these

All closed by direct measurement; the reasoning is in `out-nccl-2node/summary.md` § 2.2, *"What the data already excludes"*.

- **CPU model and frequency governor.** Both clusters are Xeon Platinum 8570 on `performance`, read first-hand on node5501. This was the cheapest outstanding hypothesis and it is now closed. (Deep C-state configuration is still unread on either side — the one residue, and a footnote rather than a step.)
- **IOMMU state.** The clusters differ — Ubuntu `iommu=pt intel_iommu=on` with 540 groups, Rocky 8 `iommu=off` with none — but both reach full GPUDirect line rate, and the difference runs the wrong way: the cluster with the IOMMU *disabled* is the slower one.
- **PCIe platform configuration.** ACS redirect, PCIe topology, Relaxed Ordering and Max Payload Size were candidates only for the old GPUDirect cap. That cap is gone and `nvidia-smi topo -m` reports **PXB** for every GPU<->rail pair on both clusters.
- **Node heterogeneity inside the Rocky cluster.** The EL8 / 4.18 vs EL10 / 6.12 split is gone; node5500 and node5501 both run 6.12.0-211.
- **A bulk bandwidth cap.** Settled twice over — by NCCL `sendrecv` and by `ib_write_bw`.

**Ground rule for everything below:** change one item on one node pair, then re-run the checks in "How to verify" before the next change, so every result stays attributable.

---

## Step 1 — Re-measure the gap on the current Rocky configuration

*One batch job. Everything else is conditional on it.*

```bash
sbatch job-nccl-2node.sh all 8      # on any two current Rocky nodes
```

**Why this comes first.** The Rocky NCCL timings the whole comparison rests on are the **2026-08-06** runs on node5500+node5502 — taken *before* node5500 was reinstalled to EL10 / 6.12 and *before* the cluster moved to `iommu=off`. Both of those changes are known to have moved the bulk GPUDirect path, which went from a 2.7x deficit to a tie. The small-message gap has not been re-measured since, so its size on the current configuration is unknown.

**Read the result:** if the 1 MiB times now match Ubuntu (all_reduce ~183 us, alltoall ~285 us at 16 GPUs), the gap closed along with the bulk path and nothing below is needed. If they are still 1.9-3.0x slower, the candidate list stands and steps 2-5 apply.

## Step 2 — Finish the `ib_write_bw` small-message sweep

*Half the data is already collected. Runs on one node. Cheapest decisive test available.*

```bash
ib_write_bw -a -n 1000 --use_cuda=0    # server and client, node5700 <-> node5701
```

This separates per-operation cost from bandwidth with NCCL out of the picture. **The Rocky 8 half is already in `../b200-nodes/notes.md`**: message rate holds at ~5.3 Mpps from 2 B to 4 KiB, about 190 ns per operation, with bandwidth scaling linearly across that range. Comparing Ubuntu's small-size message rates against those numbers attributes the gap to the IB stack or exonerates it in a single measurement — and it is the one step here that does not need two free Rocky nodes.

## Step 3 — Compare NCCL's algorithm selection

*One command per cluster. Could turn the gap into an environment-variable fix.*

```bash
NCCL_DEBUG=INFO <run script> 2>&1 | grep -iE 'algo|proto|NVLS|channels'
```

Run on **both** clusters and compare channel count, protocol (LL / LL128 / Simple) and algorithm. If Rocky 8 selects differently, the gap is a tuning problem, not a reinstall — worth knowing before touching MOFED in step 4.

**Compare `all_reduce`, `alltoall` and `scatter` specifically** — the three that differ by more than 10% at 16 GiB. A different algorithm or protocol choice on any of them would explain that collective's gap directly, and `scatter` is the most likely of the three to be a pure selection difference, since it is the one where Rocky 8 is *faster* and the only one whose gap is not per-operation cost. Ubuntu plateaus at ~290 GB/s where Rocky 8 reaches 327; if the two pick different channel counts for root-anchored patterns, that is the answer and `NCCL_MIN_NCHANNELS` fixes it on the Ubuntu side.

The same command on one node also confirms whether `all_reduce` is using **NVLS** (in-NVSwitch reduction), the likely reason it reaches 93% of the NVLink ceiling while the collectives it is built from sit at 76-77%. A cluster *not* selecting NVLS has a real and recoverable difference.

## Step 4 — Downgrade MOFED to 25.10

*The leading hypothesis, and the first item that changes anything.*

Target `OFED-internal-25.10-1.7.1.413`, as on the Ubuntu nodes; Rocky 8 currently runs 26.04-0.8.6.

**Why.** The verbs provider sets per-work-request posting cost while leaving bulk streaming untouched — exactly the measured shape — and the single-node control shows the cost is not in the launch path. With the hardware-side alternatives now closed, it is the leading candidate by a wider margin than before. Note the direction: this is a *downgrade*, because the newer stack is the one showing the higher per-operation cost, so the aim is to test for a regression.

**Hold the HCA firmware constant.** Rocky 8 runs firmware **28.49.1120** against Ubuntu's **28.47.2526**, and MOFED and firmware usually move together. If a MOFED downgrade also rolls firmware, the test stops being attributable — see step 5.

It comes after steps 1-3 because those are free or read-only and this one is not.

## Step 5 — Remaining stack differences, if the gap is still open

*Only if MOFED alone does not close it.*

1. **HCA firmware — 28.47.2526 (Ubuntu) vs 28.49.1120 (Rocky 8).** Firmware sets doorbell and completion handling, which is per-operation cost by definition, and it moves in the same direction as MOFED: Rocky 8 runs the newer one and is slower. Test it separately from step 4 so the two stay attributable.
2. **NVIDIA driver / CUDA → 570.211.01 + 12.9** (Rocky 8 runs 590.48.01 + 13.1). Affects kernel launch and GDR registration, both per-operation. Weaker than MOFED because it should also touch the `sendrecv` path, and that path is clean. Caveat: r570 caps CUDA at 12.8, so anything built against CUDA 13 must be rebuilt.
3. **Kernel — 6.8 (Ubuntu 24.04) vs 6.12 (EL10).** No longer a heterogeneity problem *within* the Rocky cluster, but still a difference between clusters, and the RDMA and DMABUF paths did change between those versions. The weakest of the three, and the most disruptive to change.

Nothing needs installing for the benchmark itself: `perftest`, `rdma-core` and the NCCL stack are present on both clusters.

---

## How to verify

After each change on Rocky 8, re-run the collectives that carry the signal and compare against the Ubuntu targets:

```bash
# the two that should move (targets, 16 GPUs, Ubuntu):
#   all_reduce  183 us at 1 MiB    268.4 GB/s at 16 GiB
#   alltoall    285 us at 1 MiB     55.4 GB/s at 16 GiB
./run-nccl-2node.sh allreduce,alltoall 8

# the two controls that should NOT move (already tied on both clusters):
#   sendrecv    50 us at 1 MiB      48.8 GB/s at 16 GiB
#   gather      47 us at 1 MiB      92.9 GB/s at 16 GiB
./run-nccl-2node.sh sendrecv,gather 8
```

A change that moves `all_reduce` and `alltoall` while leaving `sendrecv` and `gather` flat is addressing the per-operation cost. A change that moves all four is doing something else — most likely disturbing the bulk path, which is currently tied and should stay that way.

## Settings already in place

These are in `run-nccl-2node.sh` and are not things to change — listed so they are not re-derived:

- `NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15` — pin the 8 NDR rails explicitly. Letting NCCL choose works at 1 GPU/node but **fails to connect at 8 GPUs/node**.
- `NCCL_NET_GDR_LEVEL=2` — keep GPUDirect on the data path.
- Bootstrap MPI over TCP on a pinned interface (`--mca pml ob1 --mca btl tcp,self`, `--mca btl_tcp_if_include <iface>`, `--mca oob_tcp_if_include <iface>`) and disable UCC/hcoll. MPI only exchanges the NCCL unique-id; the data path is NCCL.
- Build nccl-tests against the CUDA flavour the **driver** supports (12.9 here, since r570 caps at CUDA 12.8).

If the Rocky 8 nodes are ever run **without Slurm** like these ones, they will also need passwordless ssh both ways and `memlock unlimited` in non-interactive ssh sessions (`/etc/security/limits.conf` + `UsePAM yes`).

## Loose ends

- **Deep C-states.** The governor is settled but idle-state configuration is unread on both clusters: `cpupower idle-info` on one node of each. The only residue of what used to be the top hypothesis, and cheap to close out.
- **Record more in the run output.** `run-nccl-1node.sh` and `run-nccl-2node.sh` write hostname, date, OS, kernel and CUDA flavour. Adding `cat /proc/cmdline`, `nvidia-smi --query-gpu=driver_version --format=csv`, `nvidia-smi nvlink --status`, `nvidia-smi topo -m` and the governor path is one line each. The Rocky configuration is now known first-hand, but it took a dedicated Slurm job to get it — with these lines every run would carry its own provenance, and a node with degraded NVLink links would show up as more than an unexplained low busbw.
- **Repeat the 1 MiB single-node sweep on both clusters** to put an error bar on the residual 1.10x. One `run-nccl-1node.sh` per node, a few minutes each. Every individual collective's difference sits inside the Rocky 8 node-to-node spread; only the 9-of-9 consistency of the sign argues it is real.
- **The one Ubuntu-side item:** `NCCL_MIN_NCHANNELS` / `NCCL_PROTO` sweeps for the large-message `scatter` plateau (290 vs 325-339 GB/s) — an 11.1% difference and the only one of the three above 10% where Rocky 8 leads. Start from the `NCCL_DEBUG=INFO` comparison in step 3: if the two clusters pick different channel counts for root-anchored patterns, the sweep is just confirming it. Worth doing, but after the small-message work — scatter rarely bottlenecks training, and the other two collectives are the ones that do.

---

Neither root cause is established. The order above is by evidence and cost, not by certainty: step 1 establishes whether there is still anything to explain, steps 2-3 are free or read-only, step 4 is the first that changes anything.
