# NCCL notes for the admins — suggested checks and fixes

Action list drawn from the two NCCL summaries on the Ubuntu B200 nodes. **The evidence is in those files; this one holds only what to do about it.**

- `out-nccl-1node/summary.md` — 8 GPUs, one node, NVLink 5 / NVSwitch
- `out-nccl-2node/summary.md` — 16 GPUs, two nodes, InfiniBand + GPUDirect RDMA
- Rocky 8 counterparts: `../b200-nodes/out-nccl-1node/summary.md`, `../b200-nodes/out-nccl-2node/summary.md`, `../b200-nodes/notes.md`

**The two cases give very different answers, and that contrast is the main result.** Within a node the clusters are indistinguishable; across the network Rocky 8 pays 1.9-3.0x more per operation and, on 2026-07-13 data, 2.7x on the bulk GPUDirect path. Since the single-node path exercises the GPUs, driver, CUDA and NCCL host setup but not the verbs stack or the proxy thread, the single-node tie is what localises both 2-node deficits. Section 1 is common to both; sections 2 and 3 are what each case needs on its own.

**node5700 / node5701 now serve as a known-good reference to diff against.** Nothing below needs doing on the Ubuntu nodes, with one exception noted in 3.4.

---

## 1. Common to both cases

These apply regardless of node count. The first is the cheapest item on the whole list and the only candidate that sits in *both* data paths.

### 1.1 Read the CPU frequency governor and C-state configuration

On node5500-5502, read-only:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor    # performance ?
cpupower idle-info | head -20                                 # deep C-states enabled ?
```

Why it matters in both cases: launch and completion handling run on the host CPU, and NCCL's proxy thread posts every RDMA operation there too. A `powersave` governor or deep C-states add a fixed cost per operation and nothing to a streaming transfer — precisely the measured shape in both summaries (small messages penalised, large messages untouched). The Ubuntu nodes run `performance`; the governor on node5500-5502 has not been read.

Setting `performance` where it is not already set is low-risk and helps every latency-sensitive workload, independent of anything in this comparison.

### 1.2 Record more in the run output

Several questions in both summaries cannot be settled from the archived runs because the information was never captured. `run-nccl-1node.sh` and `run-nccl-2node.sh` currently write hostname, date, OS, kernel and the CUDA build flavour. Worth adding to the header:

```bash
cat /proc/cmdline                                              # iommu=off ? iommu=pt ?
nvidia-smi --query-gpu=driver_version --format=csv             # driver version
nvidia-smi nvlink --status                                     # degraded/inactive links
nvidia-smi topo -m                                             # PXB vs NODE/SYS
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor       # governor
```

Each is one line. Their absence is why the IOMMU state on Rocky 8 is an open question (3.1), why the driver versions had to be sourced from `../b200-nodes/notes.md` rather than from the runs, and why a node running with degraded NVLink links would show up only as an unexplained low busbw.

### 1.3 Compare NCCL's own algorithm selection

```bash
NCCL_DEBUG=INFO <run script> 2>&1 | grep -iE 'algo|proto|NVLS|channels'
```

One command, two payoffs. **Across two nodes**, run it on *both* clusters and compare channel count, protocol (LL / LL128 / Simple) and algorithm: if Rocky 8 selects differently, the per-operation gap is a tuning problem fixable with environment variables rather than a reinstall — worth checking before touching MOFED. **On one node**, it confirms whether `all_reduce` is using NVLS (in-NVSwitch reduction), which is the likely reason it reaches 93% of the NVLink ceiling while the collectives it is built from sit at 76-77%. If a cluster is *not* selecting NVLS, that is a real and recoverable difference.

### 1.4 `hypercube` fails validation everywhere

It fails on all five nodes across both clusters and both node counts — `Out of bounds values : 16 FAILED` at 8 GPUs, `32 FAILED` at 16 GPUs, with large `#wrong` counts at every message size. It is a property of the test, not of either cluster, and its bandwidth numbers cannot be used while validation fails. Its non-zero exit also terminates the mpirun job at the end of every sweep. Worth dropping it from the default collective list or marking it expected-to-fail.

### 1.5 Change one thing at a time

For anything in section 3: change one item on one node pair, then re-run the perftest triplet and `run-nccl-2node.sh all 8` before the next change, so every result stays attributable.

---

## 2. Single node (NVLink / NVSwitch)

**Nothing here needs fixing on either cluster.** All five nodes perform to specification, every collective lands in the same 73-93% band of the 900 GB/s NVLink ceiling, and the two clusters agree to within 2.8% at large messages — smaller than the spread between nodes of the same cluster. The value of this run is as a **control**, and that is what changes the section 3 list:

| 2-node deficit | What the single-node tie establishes |
|----------------|--------------------------------------|
| Bulk GPUDirect cap | **Narrowed to the NIC-to-GPU PCIe leg.** Intra-node `sendrecv` is a tie (655.9 vs 674.8 GB/s), so GPU memory streams at full rate on Rocky 8. The PCIe and BIOS items in 3.3 remain the ones to check; nothing about the GPUs themselves needs investigating. |
| Per-operation cost | **Mostly network-stack, not launch-path.** 1.10x within a node against 2.23x across two. Raises the priority of the MOFED item (3.2) and lowers that of the driver/CUDA item. |

Two checks are specific to this case:

- **Confirm the NVLS selection** for `all_reduce` — see 1.3.
- **Repeat the 1 MiB sweep on both clusters** to put an error bar on the residual 1.10x. One `run-nccl-1node.sh` per node, a few minutes each. Every individual collective's difference sits inside the Rocky 8 node-to-node spread; only the 9-of-9 consistency of the sign argues it is real. This is the cheapest way to settle it.

---

## 3. Two nodes (InfiniBand)

Two independent deficits separate the clusters here.

| # | Deficit | Size | Where it shows | Rocky 8 data from |
|---|---------|------|----------------|-------------------|
| 1 | GPUDirect bulk cap | **2.7x** on NIC-reads-from-GPU (147.6 vs 395.5 Gb/s) | 1 GPU/node NCCL: 12.7 vs 48.7 GB/s | **2026-07-13 — may be stale, verify first** |
| 2 | Per-operation cost | **1.9-3.0x** in time at 1 MiB | every collective NCCL splits across its 8 channels | 2026-08-06 |

### 3.1 Before anything else: confirm the current state

**Deficit 1 rests on 2026-07-13 measurements and may no longer exist.** Every Rocky 8 figure for it dates from that day. The Rocky 8 runs used everywhere else are from **2026-08-06**, are all 8 GPUs/node, and show **no** bulk deficit (sendrecv 47.7-49.7 GB/s across the three pairs, matching Ubuntu's 48.8). Nothing in between was recorded. Deficit 2, by contrast, is measured on the 2026-08-06 data and stands.

There is also an **open question about the IOMMU state**. `../b200-nodes/notes.md` (2026-07-13) records `iommu=pt intel_iommu=on` with 540 groups; it has since been suggested the Rocky 8 nodes now run `iommu=off`. That could not be verified from node5700. **The two cases lead to opposite advice:**

- Still `iommu=pt intel_iommu=on` → the `iommu=off` diagnostic in 3.3 is worth doing.
- Already `iommu=off` and still below line rate → IOMMU is ruled out on both clusters, skip that diagnostic, and the search narrows to MOFED, BIOS/ACS state and PCIe topology.

For reference, the Ubuntu nodes run `iommu=pt intel_iommu=on` (540 groups, HCA and GPU in separate groups) and still reach line rate — IOMMU-on is compatible with full GPUDirect bandwidth on this hardware.

Three commands settle all of it, on any Rocky 8 node:

```bash
cat /proc/cmdline                      # iommu=off ? iommu=pt ?
ls /sys/kernel/iommu_groups | wc -l    # 0 => IOMMU off; ~540 => on
sbatch job-nccl-2node.sh sendrecv 1    # ~12.7 GB/s => cap persists;
                                       # ~48 GB/s  => deficit 1 is gone
```

If that last run comes back near 48 GB/s, **stop: deficit 1 no longer exists** and only the deficit-2 items are worth pursuing — the governor check in 1.1, the MOFED downgrade in 3.2, and the `NCCL_DEBUG=INFO` comparison in 1.3.

### 3.2 Libraries and drivers

1. **MOFED / rdma-core → 25.10** (`OFED-internal-25.10-1.7.1.413`, as on the Ubuntu nodes; Rocky 8 currently runs 26.04-0.8.6). The **leading hypothesis** for deficit 2: the verbs provider sets per-work-request posting cost while leaving bulk streaming untouched, which is the measured shape, and the single-node control shows the cost is not in the launch path. Not the first thing to *do* — the governor check in 1.1 is free — but the first thing to *change*. Note this is a downgrade: the newer stack is the one showing the higher per-operation cost, so the aim is to test for a regression.
2. **NVIDIA driver → 570.211.01** (Rocky 8 runs 590.48.01), only if MOFED alone does not close the gap. Lower priority than it looks: the single-node run exercises this layer and shows only a 1.10x effect. Caveat: r570 caps CUDA at 12.8, so anything built against CUDA 13 must be rebuilt.
3. **Align node5500's kernel** (EL8 / 4.18) with node5502 (EL10 / 6.12). Not a suspected cause — all three Rocky pairs measure the same — but it removes a variable.

Nothing needs installing for the benchmark itself: `perftest`, `rdma-core` and the NCCL stack are present on both clusters.

### 3.3 System configuration

**For deficit 1 (the GPUDirect bulk path)** — this is set at the platform level, so these are the items that move it:

- **`iommu=off` as a diagnostic on one Rocky node.** One cmdline edit plus a reboot, reversible, and decisive: if GPU-read bandwidth jumps from 147.6 Gb/s toward ~395, the cause is the IOMMU/ACS interaction on that platform. The mechanism is real — Linux enables ACS on downstream ports when the IOMMU is on, and ACS redirect sends peer-to-peer TLPs up to the root complex instead of straight across the switch. This does not contradict the finding that IOMMU-on is compatible with line rate: what is excluded is IOMMU-on as the explanation of the *per-operation* gap, since both clusters boot the identical setting and only one is slow.
- **Then prefer a targeted fix in production** — see 3.6. node5502 already carries `pci=disable_acs_redir=pci:1000:c030` and was still capped, which suggests that mask missed the switch ports actually in its GPU↔NIC path.
- **PCIe Relaxed Ordering** — check it is enabled in BIOS. An NVIDIA-recommended setting for GPUDirect; disabling it degrades NIC-reads-from-GPU specifically, matching the measured read/write asymmetry (147.6 read vs 286.6 write) better than any symmetric explanation.
- **PCIe topology and ACS state** — compare `nvidia-smi topo -m` and root `lspci -vvv` ACSCtl bits on the GPU↔NIC path against node5700. node5700 reports **PXB** for every GPU↔rail pair, `lspci -t` confirms HCA and GPU under the same switch, and it does *not* carry `pci=disable_acs_redir` — so the kernel workaround is not the differentiator, and the BIOS/firmware state is what to inspect. If a Rocky node reports `NODE`/`SYS` where node5700 reports `PXB`, the GPU and its rail are not under a common switch, which alone would explain the cap.
- **PCIe Max Payload Size / Max Read Request** on the HCA and GPU bridges — a smaller MPS means more TLPs per byte, again asymmetric toward reads. Diff against node5700.

**For deficit 2 (per-operation cost)**: the governor and C-states in 1.1 come first. Beyond that, report (do not change) the **CPU model** and **HCA firmware** on node5500-5502 — neither is verifiable from node5700 and both could matter for a per-operation cost.

If the Rocky 8 nodes are ever run **without Slurm** like these ones, they will also need passwordless ssh both ways and `memlock unlimited` in non-interactive ssh sessions (`/etc/security/limits.conf` + `UsePAM yes`).

### 3.4 Application settings

Application settings **do not move deficit 1** — the GPUDirect bulk path is set by platform configuration. They matter for getting the most out of whatever the platform provides. These are already in `run-nccl-2node.sh`:

- `NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15` — pin the 8 NDR rails explicitly. Letting NCCL choose works at 1 GPU/node but **fails to connect at 8 GPUs/node**.
- `NCCL_NET_GDR_LEVEL=2` — keep GPUDirect on the data path.
- Bootstrap MPI over TCP on a pinned interface (`--mca pml ob1 --mca btl tcp,self`, `--mca btl_tcp_if_include <iface>`, `--mca oob_tcp_if_include <iface>`) and disable UCC/hcoll. MPI only exchanges the NCCL unique-id; the data path is NCCL.
- Build nccl-tests against the CUDA flavour the **driver** supports (12.9 here, since r570 caps at CUDA 12.8).

**The one item that applies to the Ubuntu nodes:** `NCCL_MIN_NCHANNELS` / `NCCL_PROTO` sweeps for the large-message `scatter` plateau (290 vs 325-339 GB/s) — the single collective where Rocky 8 is ahead. Low priority: scatter rarely bottlenecks training.

### 3.5 How to verify

After each change on Rocky 8, re-run the two tests that isolate the deficits and compare against the Ubuntu targets:

```bash
# deficit 1 — GPUDirect bulk path (target: ~395 Gb/s read, ~380 Gb/s write)
ssh <nodeB> 'ib_write_bw -d mlx5_4 --report_gbits -s 67108864 -n 200'
ib_write_bw -d mlx5_4 --use_cuda=0 --report_gbits -s 67108864 -n 200 <nodeB>

# deficit 2 — per-operation cost (target: all_reduce ~183 us at 1 MiB, 16 GPUs)
./run-nccl-2node.sh allreduce 8
```

A controlled test would separate the remaining deficit-2 candidates outright: an `ib_write_bw` small-message sweep (many small ops vs one large op) on both clusters isolates the IB stack from everything above it.

### 3.6 Keeping IOMMU on while disabling ACS redirect

`iommu=off` is the broadest lever, and it is **not** the only way to stop ACS redirect from routing peer-to-peer traffic through the root complex. All three options below leave the IOMMU fully on.

**1. Kernel command line (persistent, targeted).**

```
pci=disable_acs_redir=pci:1000:c030      # by vendor:device
pci=disable_acs_redir=0000:17:02.0       # or by BDF, ';'-separated
```

Clears the P2P Request Redirect / Completion Redirect / Upstream Forwarding bits. **The devices to name are the downstream ports of the switch between the GPU and its HCA — not the GPU or the NIC.** node5502 already carries this for `pci:1000:c030` and still measures below line rate, so extending the mask to the bridges actually in its GPU↔NIC path is the next step; on node5700 that path is the switch at `[17-1b]` (HCA `0000:18:00.0`, GPU0 `0000:1b:00.0`).

**2. BIOS.** Most server BIOSes expose "PCIe ACS" / "ACS Enable". Disabling it there means the capability is never enabled at boot, so the kernel has nothing to enforce and the IOMMU stays on. Cleanest production option where it exists.

**3. `setpci` at runtime (no reboot, for A/B testing).**

```bash
setpci -s <bridge_BDF> ECAP_ACS+6.w=0000   # clear ACS control reg
```

Per bridge, as root; does **not** survive reboot or PCIe hotplug — ideal for a quick test on one node: measure `ib_write_bw --use_cuda`, clear the bits, measure again.

**Verify what is actually set** (root required):

```bash
lspci -vvv -s <bridge_BDF> | grep -A2 'Access Control Services'
```

`ACSCap:` shows what the hardware supports, `ACSCtl:` what is enabled. For P2P you want **`RR-` and `CR-`**. This is the one measurement that could not be taken on node5700 — it needs root — so whether ACS redirect is active there remains unknown, even though the bandwidth shows it is not costing anything.

Two things to weigh:

- Disabling ACS redirect **merges the affected devices into one IOMMU group**, trading some device isolation for P2P bandwidth. Not a concern for bare-metal HPC; worth weighing if those nodes ever host VMs or VFIO passthrough.
- Prefer the three in-tree options above; the out-of-tree `pcie_acs_override=` patch is aimed at VFIO passthrough rather than production HPC nodes.

Note that **`iommu=pt` is a different knob**: it makes host DMA use identity mapping to cut translation cost, but it does *not* clear ACS redirect. Both clusters already run `iommu=pt`, so it is not a substitute for any of the above.

---

## Priority

1. **CPU governor and C-states** (1.1) — free, read-only, in both data paths, and may explain deficit 2 outright.
2. **Confirm deficit 1 still exists** (3.1) — three commands; if it is gone, most of section 3 falls away.
3. **`NCCL_DEBUG=INFO` on both clusters** (1.3) — one command, and a different protocol selection would make deficit 2 an environment-variable fix.
4. **PCIe / BIOS diff against node5700** (3.3) — the larger gap, and there is now a working reference to diff against.
5. **MOFED downgrade** (3.2) — the leading hypothesis for deficit 2, but the first item that changes anything, so it comes after the free checks.

Neither root cause is established; this list is ordered by evidence and cost, not by certainty.
