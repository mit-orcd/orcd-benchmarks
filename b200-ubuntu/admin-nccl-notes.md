# NCCL notes for the admins — what to test, in order

Action list drawn from the two NCCL summaries on the Ubuntu B200 nodes. **The evidence is in those files; this one holds only what to do about it.**

- `out-nccl-1node/summary.md` — 8 GPUs, one node, NVLink 5 / NVSwitch
- `out-nccl-2node/summary.md` — 16 GPUs, two nodes, InfiniBand + GPUDirect RDMA
- Rocky 8 counterparts: `../b200-nodes/out-nccl-1node/summary.md`, `../b200-nodes/out-nccl-2node/summary.md`, `../b200-nodes/notes.md`

**Nothing below needs doing on node5700 / node5701.** They perform to specification and now serve as a known-good reference to diff against. The one Ubuntu-side item is in "Loose ends" at the end.

---

## What there is to explain

Two things separate Rocky 8 from Ubuntu, and only across the network. Both are named here exactly as the summary sections they come from, so each row can be looked up:

| What | Size | Read the evidence in |
|------|------|----------------------|
| **Bulk GPUDirect path** — NIC reads from GPU, 147.6 vs 395.5 Gb/s | **2.7x** | `out-nccl-2node/summary.md` § 4.1, *"Bulk GPUDirect path, measured directly"* — and § 4.2, *"For the bulk GPUDirect difference…"* for the candidates. **Rocky 8 data is 2026-07-13 and may be stale; step 2 checks.** |
| **Small messages** — time at 1 MiB, per-operation cost | **1.9-3.0x** | `out-nccl-2node/summary.md` § 4.1, *"Small messages"* — and § 4.2, *"The shape of the gap constrains the explanation"* for the candidates. Rocky 8 data is 2026-08-06. |

**Within a single node neither one appears.** `out-nccl-1node/summary.md` § 4.1 puts every collective in the same 73-93% band of the NVLink ceiling on both clusters, with large messages within 2.8%; its § 4.2 spells out what that rules out. Since the single-node path exercises the GPUs, driver, CUDA and NCCL host setup but *not* the verbs stack or the proxy thread, that tie is what localises both rows above: the bulk GPUDirect path is confined to the NIC-to-GPU PCIe leg rather than the GPUs, and the small-message cost is mostly network-stack rather than launch-path (1.10x within a node against 2.23x across two).

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

## Step 2 — Confirm the bulk GPUDirect path is still capped

*Three commands. If it is gone, most of this list falls away.*

```bash
cat /proc/cmdline                      # iommu=off ? iommu=pt ?
ls /sys/kernel/iommu_groups | wc -l    # 0 => IOMMU off; ~540 => on
sbatch job-nccl-2node.sh sendrecv 1    # ~12.7 GB/s => cap persists
                                       # ~48 GB/s  => the cap is gone
```

**Why.** Every Rocky 8 figure for the bulk GPUDirect path dates from 2026-07-13. The runs used everywhere else are from 2026-08-06, are all 8 GPUs/node, and show **no** cap (sendrecv 47.7-49.7 GB/s across the three pairs, matching Ubuntu's 48.8). Nothing in between was recorded. The small-message gap, by contrast, is measured on the 2026-08-06 data and stands regardless.

**The IOMMU line matters too.** `../b200-nodes/notes.md` records `iommu=pt intel_iommu=on` with 540 groups; it has since been suggested the nodes now run `iommu=off`. The two cases lead to opposite advice in step 4. For reference, the Ubuntu nodes run `iommu=pt intel_iommu=on` (540 groups, HCA and GPU in separate groups) and still reach line rate — IOMMU-on is compatible with full GPUDirect bandwidth on this hardware.

**Read the result:**

- **~48 GB/s** → stop pursuing the bulk GPUDirect path. Skip step 4; go to step 3, then 5.
- **~12.7 GB/s** → the cap is live. Continue through all steps.

## Step 3 — Compare NCCL's algorithm selection

*One command per cluster. Could turn the small-message gap into an environment-variable fix.*

```bash
NCCL_DEBUG=INFO <run script> 2>&1 | grep -iE 'algo|proto|NVLS|channels'
```

Run on **both** clusters and compare channel count, protocol (LL / LL128 / Simple) and algorithm. If Rocky 8 selects differently, the small-message gap is a tuning problem, not a reinstall — worth knowing before touching MOFED in step 5.

The same command on one node also confirms whether `all_reduce` is using **NVLS** (in-NVSwitch reduction), the likely reason it reaches 93% of the NVLink ceiling while the collectives it is built from sit at 76-77%. A cluster *not* selecting NVLS has a real and recoverable difference.

## Step 4 — Diff PCIe and BIOS state against node5700

*Only if step 2 showed the cap persists. This is where the bulk GPUDirect path is lost.*

The bulk GPUDirect path is set at the platform level, so these are the items that move it. Compare each against node5700:

- **PCIe topology** — `nvidia-smi topo -m`. node5700 reports **PXB** for every GPU↔rail pair. If a Rocky node reports `NODE`/`SYS`, its GPU and rail are not under a common switch and peer-to-peer goes through the host bridge, which alone explains the cap.
- **ACS state** — root `lspci -vvv` ACSCtl bits on the GPU↔NIC path. node5700 does *not* carry `pci=disable_acs_redir`, so the kernel workaround is not the differentiator and the BIOS/firmware state is what to inspect. node5502 already carries `pci=disable_acs_redir=pci:1000:c030` and was still capped, which suggests that mask missed the switch ports actually in its path.
- **PCIe Relaxed Ordering** — check it is enabled in BIOS. NVIDIA-recommended for GPUDirect; disabling it degrades NIC-reads-from-GPU specifically, matching the measured read/write asymmetry (147.6 read vs 286.6 write) better than any symmetric explanation.
- **PCIe Max Payload Size / Max Read Request** on the HCA and GPU bridges — a smaller MPS means more TLPs per byte, again asymmetric toward reads.

**The `iommu=off` diagnostic.** One cmdline edit plus a reboot on one node, reversible, and decisive: if GPU-read bandwidth jumps from 147.6 Gb/s toward ~395, the cause is the IOMMU/ACS interaction on that platform. The mechanism is real — Linux enables ACS on downstream ports when the IOMMU is on, and ACS redirect sends peer-to-peer TLPs up to the root complex instead of straight across the switch. Worth doing **only if step 2 found `iommu=pt` still set**; if the nodes are already `iommu=off` and still capped, IOMMU is ruled out and the search narrows to the bullets above plus MOFED.

Do not leave `iommu=off` in production — see "Keeping IOMMU on while disabling ACS redirect" below for the targeted fixes.

## Step 5 — Downgrade MOFED to 25.10

*The leading hypothesis for the small-message gap, and the first item that changes anything.*

Target `OFED-internal-25.10-1.7.1.413`, as on the Ubuntu nodes; Rocky 8 currently runs 26.04-0.8.6.

**Why.** The verbs provider sets per-work-request posting cost while leaving bulk streaming untouched — exactly the measured shape — and the single-node control shows the cost is not in the launch path. Note this is a *downgrade*: the newer stack is the one showing the higher per-operation cost, so the aim is to test for a regression.

It comes after steps 1-4 because those are free and this one is not.

## Step 6 — Align driver and kernel, if the gap is still open

*Only if MOFED alone does not close it.*

1. **NVIDIA driver → 570.211.01** (Rocky 8 runs 590.48.01). Lower priority than it looks: the single-node run exercises this layer and shows only a 1.10x effect. Caveat: r570 caps CUDA at 12.8, so anything built against CUDA 13 must be rebuilt.
2. **node5500's kernel** (EL8 / 4.18) → align with node5502 (EL10 / 6.12). Not a suspected cause — all three Rocky pairs measure the same — but it removes a variable.
3. **Report (do not change) the CPU model and HCA firmware** on node5500-5502. Neither is verifiable from node5700 and both could matter for a per-operation cost.

Nothing needs installing for the benchmark itself: `perftest`, `rdma-core` and the NCCL stack are present on both clusters.

---

## How to verify

After each change on Rocky 8, re-run the two tests that isolate the two rows above and compare against the Ubuntu targets:

```bash
# bulk GPUDirect path (target: ~395 Gb/s read, ~380 Gb/s write)
ssh <nodeB> 'ib_write_bw -d mlx5_4 --report_gbits -s 67108864 -n 200'
ib_write_bw -d mlx5_4 --use_cuda=0 --report_gbits -s 67108864 -n 200 <nodeB>

# small messages (target: all_reduce ~183 us at 1 MiB, 16 GPUs)
./run-nccl-2node.sh allreduce 8
```

A controlled test would separate the remaining small-message candidates outright: an `ib_write_bw` small-message sweep (many small ops vs one large op) on both clusters isolates the IB stack from everything above it.

## Keeping IOMMU on while disabling ACS redirect

`iommu=off` is the broadest lever and **not** the only way to stop ACS redirect from routing peer-to-peer traffic through the root complex. All three options below leave the IOMMU fully on.

**1. Kernel command line (persistent, targeted).**

```
pci=disable_acs_redir=pci:1000:c030      # by vendor:device
pci=disable_acs_redir=0000:17:02.0       # or by BDF, ';'-separated
```

Clears the P2P Request Redirect / Completion Redirect / Upstream Forwarding bits. **The devices to name are the downstream ports of the switch between the GPU and its HCA — not the GPU or the NIC.** node5502 already carries this for `pci:1000:c030` and still measures below line rate, so extending the mask to the bridges actually in its path is the next step; on node5700 that path is the switch at `[17-1b]` (HCA `0000:18:00.0`, GPU0 `0000:1b:00.0`).

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

- Disabling ACS redirect **merges the affected devices into one IOMMU group**, trading device isolation for P2P bandwidth. Not a concern for bare-metal HPC; worth weighing if those nodes ever host VMs or VFIO passthrough.
- Prefer the three in-tree options above; the out-of-tree `pcie_acs_override=` patch is aimed at VFIO passthrough rather than production HPC nodes.

Note that **`iommu=pt` is a different knob**: it makes host DMA use identity mapping to cut translation cost, but it does *not* clear ACS redirect. Both clusters already run `iommu=pt`, so it is not a substitute for any of the above.

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
- **The one Ubuntu-side item:** `NCCL_MIN_NCHANNELS` / `NCCL_PROTO` sweeps for the large-message `scatter` plateau (290 vs 325-339 GB/s), the single collective where Rocky 8 is ahead. Low priority — scatter rarely bottlenecks training.

---

Neither root cause is established. The order above is by evidence and cost, not by certainty: steps 1-4 are free or read-only, step 5 is the first that changes anything.
