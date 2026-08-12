# NCCL notes for the admins — what to test, in order

Action list drawn from the two NCCL summaries on the Ubuntu B200 nodes. **The evidence is in those files; this one holds only what to do about it.**

- `out-nccl-1node/summary.md` — 8 GPUs, one node, NVLink 5 / NVSwitch
- `out-nccl-2node/summary.md` — 16 GPUs, two nodes, InfiniBand + GPUDirect RDMA
- Rocky 8 counterparts: `../b200-nodes/out-nccl-1node/summary.md`, `../b200-nodes/out-nccl-2node/summary.md`, `../b200-nodes/notes.md`

**Nothing below needs doing on node5700 / node5701.** They perform to specification and now serve as a known-good reference to diff against. The one Ubuntu-side item is in "Loose ends" at the end.

---

## What there is to explain

One thing separates Rocky 8 from Ubuntu, and only across the network. It is named here exactly as the summary section it comes from, so it can be looked up:

| What | Size | Read the evidence in |
|------|------|----------------------|
| **Small messages** — time at 1 MiB, per-operation cost | **1.9-3.0x** | `out-nccl-2node/summary.md` § 2.1, *"Small messages"* — and § 2.2, *"The shape of the gap constrains the explanation"* for the candidates. Rocky 8 data is 2026-08-06. |

Bulk transfer is **not** a difference: NCCL `sendrecv` at 16 GPUs converges to 48.4 GB/s per pair on Rocky 8 against 48.8 on Ubuntu, at 98% of one rail on both.

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

**Ground rule for everything below:** change one item on one node pair, then re-run the checks in "How to verify" before the next change, so every result stays attributable.

---

## Step 1 — Read the CPU governor and C-states

*Free, read-only, no reboot. Do this first.*

On node5500-5502:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor    # performance ?
cpupower idle-info | head -20                                # deep C-states enabled ?
```

**Why first.** It is the only candidate sitting in *both* data paths, and it costs nothing to check. Launch and completion handling run on the host CPU, and NCCL's proxy thread posts every RDMA operation there too. A `powersave` governor or deep C-states add a fixed cost per operation and nothing to a streaming transfer — precisely the measured shape (small messages penalised, large untouched). The Ubuntu nodes run `performance`; the Rocky governor has never been read.

**If it is not `performance`,** set it. Low-risk, helps every latency-sensitive workload independent of anything here, and it may explain the small-message gap outright.

## Step 2 — Compare NCCL's algorithm selection

*One command per cluster. Could turn the small-message gap into an environment-variable fix.*

```bash
NCCL_DEBUG=INFO <run script> 2>&1 | grep -iE 'algo|proto|NVLS|channels'
```

Run on **both** clusters and compare channel count, protocol (LL / LL128 / Simple) and algorithm. If Rocky 8 selects differently, the small-message gap is a tuning problem, not a reinstall — worth knowing before touching MOFED in step 3.

**Compare `all_reduce`, `alltoall` and `scatter` specifically** — the three that differ by more than 10% at 16 GiB. A different algorithm or protocol choice on any of them would explain that collective's gap directly, and `scatter` is the most likely of the three to be a pure selection difference, since it is the one where Rocky 8 is *faster* and the only one whose gap is not per-operation cost. Ubuntu plateaus at ~290 GB/s where Rocky 8 reaches 327; if the two pick different channel counts for root-anchored patterns, that is the answer and `NCCL_MIN_NCHANNELS` fixes it on the Ubuntu side.

The same command on one node also confirms whether `all_reduce` is using **NVLS** (in-NVSwitch reduction), the likely reason it reaches 93% of the NVLink ceiling while the collectives it is built from sit at 76-77%. A cluster *not* selecting NVLS has a real and recoverable difference.

## Step 3 — Downgrade MOFED to 25.10

*The leading hypothesis for the small-message gap, and the first item that changes anything.*

Target `OFED-internal-25.10-1.7.1.413`, as on the Ubuntu nodes; Rocky 8 currently runs 26.04-0.8.6.

**Why.** The verbs provider sets per-work-request posting cost while leaving bulk streaming untouched — exactly the measured shape — and the single-node control shows the cost is not in the launch path. Note this is a *downgrade*: the newer stack is the one showing the higher per-operation cost, so the aim is to test for a regression.

It comes after steps 1-2 because those are free and this one is not.

## Step 4 — Align driver and kernel, if the gap is still open

*Only if MOFED alone does not close it.*

1. **NVIDIA driver → 570.211.01** (Rocky 8 runs 590.48.01). Lower priority than it looks: the single-node run exercises this layer and shows only a 1.10x effect. Caveat: r570 caps CUDA at 12.8, so anything built against CUDA 13 must be rebuilt.
2. **node5500's kernel** (EL8 / 4.18) → align with node5502 (EL10 / 6.12). Not a suspected cause — all three Rocky pairs measure the same — but it removes a variable.
3. **Report (do not change) the CPU model and HCA firmware** on node5500-5502. Neither is verifiable from node5700 and both could matter for a per-operation cost.

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

A controlled test would separate the candidates outright: an `ib_write_bw` small-message sweep (many small ops vs one large op) on both clusters isolates the IB stack from everything above it.

## Settings already in place

These are in `run-nccl-2node.sh` and are not things to change — listed so they are not re-derived:

- `NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15` — pin the 8 NDR rails explicitly. Letting NCCL choose works at 1 GPU/node but **fails to connect at 8 GPUs/node**.
- `NCCL_NET_GDR_LEVEL=2` — keep GPUDirect on the data path.
- Bootstrap MPI over TCP on a pinned interface (`--mca pml ob1 --mca btl tcp,self`, `--mca btl_tcp_if_include <iface>`, `--mca oob_tcp_if_include <iface>`) and disable UCC/hcoll. MPI only exchanges the NCCL unique-id; the data path is NCCL.
- Build nccl-tests against the CUDA flavour the **driver** supports (12.9 here, since r570 caps at CUDA 12.8).

If the Rocky 8 nodes are ever run **without Slurm** like these ones, they will also need passwordless ssh both ways and `memlock unlimited` in non-interactive ssh sessions (`/etc/security/limits.conf` + `UsePAM yes`).

## Loose ends

- **Record more in the run output.** `run-nccl-1node.sh` and `run-nccl-2node.sh` write hostname, date, OS, kernel and CUDA flavour. Adding `cat /proc/cmdline`, `nvidia-smi --query-gpu=driver_version --format=csv`, `nvidia-smi nvlink --status`, `nvidia-smi topo -m` and the governor path is one line each. Their absence is why the Rocky IOMMU state is an open question at all, why driver versions had to come from `../b200-nodes/notes.md` rather than the runs, and why a node with degraded NVLink links would show up only as an unexplained low busbw.
- **Repeat the 1 MiB single-node sweep on both clusters** to put an error bar on the residual 1.10x. One `run-nccl-1node.sh` per node, a few minutes each. Every individual collective's difference sits inside the Rocky 8 node-to-node spread; only the 9-of-9 consistency of the sign argues it is real.
- **The one Ubuntu-side item:** `NCCL_MIN_NCHANNELS` / `NCCL_PROTO` sweeps for the large-message `scatter` plateau (290 vs 325-339 GB/s) — an 11.1% difference and the only one of the three above 10% where Rocky 8 leads. Start from the `NCCL_DEBUG=INFO` comparison in step 2: if the two clusters pick different channel counts for root-anchored patterns, the sweep is just confirming it. Worth doing, but after the small-message work — scatter rarely bottlenecks training, and the other two collectives are the ones that do.

---

Neither root cause is established. The order above is by evidence and cost, not by certainty: steps 1-2 are free or read-only, step 3 is the first that changes anything.
