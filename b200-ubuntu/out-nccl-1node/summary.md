# nccl-tests 1-node summary — Ubuntu B200 nodes

- Generated: 2026-08-10 14:52:02 — restructured 2026-08-12
- Runs: node5700 (2026-08-10 14:32), node5701 (2026-08-10 14:35)
- GPUs: 8 x NVIDIA B200 in one node (intra-node, NVLink 5 / NVSwitch — no InfiniBand on this path)
- Config: 1 thread, 1 MiB-16 GiB, 5 warmup + 20 iters
- **busbw** (bus bandwidth) is the figure of merit throughout; results are judged against this platform's own hardware ceiling (`HW max`), not against an external reference.
- **node5700/node5701 run Ubuntu 24.04; node5500-5502 run Rocky 8.** Sections 1-3 report the Ubuntu nodes on their own. The entire Ubuntu-vs-Rocky 8 comparison — what differs, which is better, and why — is consolidated in **section 4**. Full Rocky 8 set: `../b200-nodes/out-nccl-1node/summary.md`.

## 1. Results — bandwidth, and how close it gets to the hardware limit

A *collective* is one communication pattern that all 8 GPUs take part in together (e.g. `all_reduce` sums a buffer across every GPU; `broadcast` sends one GPU's buffer to all). The figure of merit is **busbw** (bus bandwidth, GB/s) at the largest message size.

`HW max` is **900 GB/s for every row**: each B200 carries 18 NVLink 5 links at 50 GB/s per direction, and all 8 GPUs are fully connected through a non-blocking NVSwitch, so every collective — ring, all-to-all or root-anchored — is limited by the same per-GPU egress into the switch. Dividing by one link's 50 GB/s gives the effective link count: how much of that egress the collective actually engages.

| Collective | GPUs | node5700 | node5701 | % of HW max | effective links (of 18) | correctness |
|------------|-----:|---------:|---------:|------------:|------------------------:|:-----------:|
| all_reduce | 8 | 841.3 | 839.1 | 93% | 16.8 | PASS |
| scatter | 8 | 746.0 | 733.5 | 82% | 14.8 | PASS |
| gather | 8 | 718.1 | 718.0 | 80% | 14.4 | PASS |
| reduce_scatter | 8 | 696.1 | 695.3 | 77% | 13.9 | PASS |
| reduce | 8 | 682.2 | 701.6 | 77% | 13.8 | PASS |
| broadcast | 8 | 681.3 | 684.9 | 76% | 13.7 | PASS |
| all_gather | 8 | 679.9 | 680.4 | 76% | 13.6 | PASS |
| alltoall | 8 | 661.3 | 660.0 | 73% | 13.2 | PASS |
| sendrecv | 8 | 655.6 | 656.2 | 73% | 13.1 | PASS |

busbw at 16 GiB, best of out-of-place / in-place, in GB/s. `% of HW max` and effective links use the mean of the two nodes.

**The two nodes agree** to within 2.8% on every collective, seven of the nine to within 0.5%, so nothing below rests on a single node.

**The headline is the narrow band.** Eight of the nine land between 73% and 82% across patterns as different as a point-to-point ring, an all-to-all shuffle and a fan-in to a single root. That clustering *is* the health result: no collective falls off a cliff, and the residual 18-27% is the ordinary gap between NVLink's signalling rate and what a real collective sustains (ring fill and drain, protocol framing, reduction kernels competing for GPU resources). There is no per-collective pathology to chase.

**`all_reduce` at 93% is the one number worth a second look** — it beats `all_gather` (76%), the pattern a Ring all_reduce is built from and therefore its ceiling. The likely explanation is **NVLS (NVLink SHARP)**, which reduces inside the NVSwitch so fewer bytes cross the wire per user byte. Unconfirmed; `NCCL_DEBUG=INFO` would settle it by naming the algorithm.

## 2. Bandwidth vs message size (GB/s)

Representative node: **node5700** (node5701 agrees to within 2.8% at every converged point; see section 1).

### all_gather

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 48.5 us | 18.9 | 48.0 us | 19.1 |
| 4 MiB | 46.5 us | 79.0 | 46.8 us | 78.5 |
| 16 MiB | 106.0 us | 138.4 | 103.4 us | 142.0 |
| 64 MiB | 141.5 us | 414.9 | 140.9 us | 416.8 |
| 256 MiB | 405.9 us | 578.6 | 403.5 us | 582.1 |
| 1 GiB | 1.52 ms | 618.0 | 1.51 ms | 623.5 |
| 4 GiB | 5.74 ms | 654.8 | 5.67 ms | 662.5 |
| 16 GiB | 22.45 ms | 669.7 | 22.11 ms | 679.9 |

### all_reduce

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 44.9 us | 40.9 | 44.2 us | 41.5 |
| 4 MiB | 58.4 us | 125.7 | 59.1 us | 124.2 |
| 16 MiB | 110.0 us | 266.9 | 108.1 us | 271.5 |
| 64 MiB | 278.6 us | 421.6 | 278.5 us | 421.7 |
| 256 MiB | 710.5 us | 661.2 | 714.2 us | 657.8 |
| 1 GiB | 2.58 ms | 728.7 | 2.58 ms | 728.7 |
| 4 GiB | 9.01 ms | 834.1 | 9.02 ms | 833.5 |
| 16 GiB | 35.73 ms | 841.3 | 35.75 ms | 841.1 |

### alltoall

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 57.3 us | 16.0 | 56.2 us | 16.3 |
| 4 MiB | 63.2 us | 58.1 | 64.0 us | 57.4 |
| 16 MiB | 70.1 us | 209.5 | 69.7 us | 210.5 |
| 64 MiB | 140.4 us | 418.2 | 142.1 us | 413.1 |
| 256 MiB | 445.9 us | 526.8 | 443.2 us | 530.0 |
| 1 GiB | 1.55 ms | 604.6 | 1.58 ms | 593.8 |
| 4 GiB | 5.81 ms | 647.0 | 5.83 ms | 644.3 |
| 16 GiB | 22.75 ms | 660.8 | 22.73 ms | 661.3 |

### broadcast

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 38.9 us | 27.0 | 38.7 us | 27.1 |
| 4 MiB | 43.2 us | 97.0 | 42.7 us | 98.2 |
| 16 MiB | 55.6 us | 301.8 | 55.7 us | 301.1 |
| 64 MiB | 135.1 us | 496.6 | 136.5 us | 491.5 |
| 256 MiB | 438.7 us | 611.8 | 439.1 us | 611.4 |
| 1 GiB | 1.64 ms | 653.2 | 1.65 ms | 651.3 |
| 4 GiB | 6.43 ms | 667.7 | 6.43 ms | 667.5 |
| 16 GiB | 25.27 ms | 679.9 | 25.22 ms | 681.3 |

### gather

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 33.6 us | 27.3 | 33.3 us | 27.6 |
| 4 MiB | 34.2 us | 107.1 | 34.1 us | 107.6 |
| 16 MiB | 35.5 us | 413.1 | 35.4 us | 415.2 |
| 64 MiB | 95.9 us | 612.4 | 96.2 us | 610.4 |
| 256 MiB | 339.8 us | 691.3 | 338.8 us | 693.2 |
| 1 GiB | 1.35 ms | 698.0 | 1.35 ms | 697.4 |
| 4 GiB | 5.24 ms | 717.0 | 5.24 ms | 716.7 |
| 16 GiB | 20.93 ms | 718.1 | 20.94 ms | 718.0 |

### reduce

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 38.8 us | 27.0 | 38.4 us | 27.3 |
| 4 MiB | 41.9 us | 100.2 | 41.6 us | 100.7 |
| 16 MiB | 53.1 us | 316.0 | 52.8 us | 317.6 |
| 64 MiB | 133.5 us | 502.7 | 133.7 us | 502.1 |
| 256 MiB | 438.6 us | 612.0 | 439.3 us | 611.0 |
| 1 GiB | 1.63 ms | 658.1 | 1.63 ms | 658.0 |
| 4 GiB | 6.37 ms | 674.6 | 6.36 ms | 675.0 |
| 16 GiB | 25.23 ms | 680.9 | 25.18 ms | 682.2 |

### reduce_scatter

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 43.8 us | 20.9 | 44.2 us | 20.8 |
| 4 MiB | 43.2 us | 85.0 | 43.3 us | 84.8 |
| 16 MiB | 102.4 us | 143.3 | 101.4 us | 144.8 |
| 64 MiB | 141.4 us | 415.3 | 141.7 us | 414.4 |
| 256 MiB | 400.8 us | 586.1 | 399.6 us | 587.7 |
| 1 GiB | 1.47 ms | 639.9 | 1.47 ms | 640.3 |
| 4 GiB | 5.53 ms | 680.1 | 5.52 ms | 680.9 |
| 16 GiB | 21.59 ms | 696.1 | 21.60 ms | 696.0 |

### scatter

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 34.5 us | 26.6 | 34.0 us | 27.0 |
| 4 MiB | 35.1 us | 104.4 | 35.2 us | 104.2 |
| 16 MiB | 36.4 us | 403.8 | 36.1 us | 406.7 |
| 64 MiB | 98.9 us | 593.9 | 99.3 us | 591.6 |
| 256 MiB | 353.1 us | 665.2 | 353.1 us | 665.2 |
| 1 GiB | 1.32 ms | 712.9 | 1.32 ms | 711.3 |
| 4 GiB | 5.07 ms | 741.2 | 5.08 ms | 740.0 |
| 16 GiB | 20.15 ms | 746.0 | 20.21 ms | 743.7 |

### sendrecv

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 34.6 us | 30.3 | 34.9 us | 30.0 |
| 4 MiB | 68.2 us | 61.5 | 67.4 us | 62.2 |
| 16 MiB | 217.7 us | 77.1 | 216.0 us | 77.7 |
| 64 MiB | 796.9 us | 84.2 | 785.6 us | 85.4 |
| 256 MiB | 813.9 us | 329.8 | 801.1 us | 335.1 |
| 1 GiB | 1.69 ms | 636.3 | 1.68 ms | 639.2 |
| 4 GiB | 6.60 ms | 651.2 | 6.71 ms | 639.9 |
| 16 GiB | 26.21 ms | 655.6 | 26.74 ms | 642.5 |

OOP = out-of-place, IP = in-place.

`sendrecv` is worth a second look: it is nearly flat from 16 MiB to 64 MiB (77 -> 84 GB/s) and then jumps 4x to 330 GB/s at 256 MiB. That step is NCCL switching protocol and channel count as the per-channel chunk grows past its thresholds, not a fabric effect; the same shape appears on all five nodes across both clusters.

## 3. Intra-node fabric

The data path here is entirely **NVLink 5 / NVSwitch**. All 8 B200s in a node are fully connected through the switch, so any GPU pair communicates at the full per-GPU link rate without traversing PCIe or the host.

| Property | Value |
|----------|-------|
| Per-GPU NVLink bandwidth | 1.8 TB/s bidirectional = **900 GB/s per direction** |
| Link structure | 18 x NVLink 5 links per GPU, 50 GB/s per direction each |
| Topology | 8 GPUs, all-to-all via NVSwitch (non-blocking) |
| GPUs seen by the run | 8 x NVIDIA B200, PCI 0x1b, 0x43, 0x52, 0x61, 0x9d, 0xc3, 0xd1, 0xdf |

Neither InfiniBand, `nvidia_peermem` nor GPUDirect RDMA is involved on this path, so nothing measured here depends on the network stack or its configuration (section 4.2).

The 900 GB/s figure and the 18-link structure are the B200 platform specification, the same basis the AICR reference (`results_b200.md`, Table 1) uses for its NVLink ceiling. They were not read back from these nodes: `nvidia-smi nvlink --status` is not captured by `run-nccl-1node.sh`. Adding it to the run script would make the ceiling self-documenting, and would also catch a node running with degraded links.

## 4. Ubuntu 24.04 vs Rocky 8

Same hardware on both sides: 8x B200 per node on NVSwitch, NCCL 2.29.2. The Ubuntu column is the mean of node5700 and node5701 (2026-08-10); the Rocky 8 column is the mean of node5500, node5501 and node5502 (2026-08-06), all five nodes running the identical sweep. Rocky 8 data: `../b200-nodes/out-nccl-1node/summary.md`.

### 4.1 What differs, and which is better

**On the intra-node path the two clusters are equivalent, with a small Ubuntu edge on small messages.** A workload confined to one node will run the same on either cluster.

**Large messages (16 GiB, converged busbw) — a tie.** Signed difference: **+** means Ubuntu is faster.

| Collective | Ubuntu busbw (GB/s) | Rocky 8 busbw (GB/s) | difference | Rocky 8 node-to-node spread |
|------------|--------------------:|---------------------:|-----------:|----------------------------:|
| reduce | 691.9 | 677.1 | +2.2% | 2.1% |
| scatter | 739.8 | 729.8 | +1.4% | 6.2% |
| all_reduce | 840.2 | 833.0 | +0.9% | 8.4% |
| reduce_scatter | 695.7 | 691.3 | +0.6% | 7.2% |
| alltoall | 660.6 | 662.4 | -0.3% | 4.7% |
| gather | 718.0 | 724.7 | -0.9% | 7.8% |
| all_gather | 680.2 | 687.1 | -1.0% | 8.1% |
| broadcast | 683.1 | 695.9 | -1.8% | 4.4% |
| sendrecv | 655.9 | 674.8 | -2.8% | 4.6% |

**Every difference is smaller than the spread between nodes of the same cluster.** The largest gap is 2.8%, while the three Rocky 8 nodes differ from each other by up to 8.4% on the same collective. Four collectives favour Ubuntu, five favour Rocky 8, and the signs alternate with no pattern. This is a tie, not a narrow win. Correctness is identical too: every collective PASSes on all five nodes.

**Small messages (time — lower is better) — a small Ubuntu advantage that decays away.** The last three columns are Rocky/Ubuntu at 1 MiB, 16 MiB and 256 MiB; above 1.00x means Ubuntu is faster.

| Collective | Ubuntu 1 MiB (us) | Rocky 8 1 MiB (us) | 1 MiB | 16 MiB | 256 MiB |
|------------|------------------:|-------------------:|------:|-------:|--------:|
| all_reduce | 44.7 | 51.7 | 1.16x | 1.02x | 1.02x |
| gather | 33.6 | 38.6 | 1.15x | 1.16x | 0.99x |
| all_gather | 48.4 | 55.7 | 1.15x | 1.02x | 1.01x |
| sendrecv | 34.6 | 38.4 | 1.11x | 0.99x | 0.98x |
| reduce | 38.6 | 43.0 | 1.11x | 1.02x | 0.97x |
| scatter | 34.4 | 37.1 | 1.08x | 0.99x | 0.98x |
| alltoall | 56.9 | 60.6 | 1.07x | 1.01x | 0.99x |
| reduce_scatter | 44.4 | 47.3 | 1.06x | 0.97x | 0.98x |
| broadcast | 40.3 | 41.0 | 1.02x | 0.98x | 0.98x |
| **mean** | | | **1.10x** | **1.02x** | **0.99x** |

**Two points.** First, at 1 MiB Ubuntu is ahead on 9 of 9 collectives, mean 1.10x — roughly 4-7 us per operation. No single row is conclusive on its own (each sits inside the 5.8-15.1% Rocky node-to-node spread), but the sign is the same in all nine while the two Ubuntu nodes agree to within 7.5%, so the effect is real, if modest. Second, it is **gone by 16 MiB and slightly reversed by 256 MiB** — the signature of a fixed cost per operation, not a bandwidth difference. `gather` is the one collective still showing 1.16x at 16 MiB, which is where it is still latency-bound (35 us).

### 4.2 Possible reasons — differences in system configuration

The clusters differ in several ways at once. What matters for reading this run is which of those differences the measurement can actually see: on one node the data path is GPU-NVSwitch-GPU, so anything belonging to the network stack is excluded by construction. Items in the 1-node data path are marked accordingly.

| Item | Ubuntu (node5700/5701) | Rocky 8 (node5500-5502) | same? | in the 1-node path? |
|------|------------------------|-------------------------|-------|---------------------|
| GPU / NVLink hardware | 8x B200, NVLink 5 / NVSwitch | 8x B200, NVLink 5 / NVSwitch | **same** | **yes** |
| NCCL | 2.29.2 | 2.29.2 | **same** | **yes** |
| **NVIDIA driver** | **570.211.01** | **590.48.01** | **differs** | **yes** |
| **CUDA (build)** | 12.9 | 13.1 | **differs** | **yes** |
| **Kernel** | 6.8.0-124 on both nodes | **4.18** (5500) / **6.12** (5502) | **differs** | **yes** (host side) |
| CPU / governor | Xeon Platinum 8570, `performance` | not verifiable from here | **unknown** | **yes** (host side) |
| **MOFED / rdma-core** | OFED-internal-25.10-1.7.1.413 | OFED-internal-26.04-0.8.6 | differs | **no** |
| GPUDirect RDMA / `nvidia_peermem` | loaded, DMABUF path | loaded, DMABUF path | same | **no** |
| IB rails | 8 x 400 Gb/s NDR | 8 x 400 Gb/s NDR | same | **no** |
| PCIe ACS / IOMMU state | `iommu=pt intel_iommu=on`, 540 groups | `iommu=pt intel_iommu=on`, 540 groups | same | **no** (NVSwitch bypasses PCIe) |

**What the measurement rules out.** The GPUs, NVLink and NVSwitch are healthy on both clusters — nine collectives within 2.8%, all in the same 73-93% band of `HW max` — and bulk GPU-to-GPU transfer is unaffected by the OS: every collective converges to the same band at 16 GiB. Neither the accelerators nor the intra-node fabric separates the two.

**What is left is the 1.10x at 1 MiB**, a fixed cost per operation on the host side. In decreasing order of plausibility:

1. **CPU frequency governor and C-states.** The most economical explanation: launch and completion handling run on the host CPU, so `powersave` or deep C-states add microseconds per operation and nothing to a streaming transfer. Ubuntu runs `performance`; the Rocky governor has not been read, so this run can neither confirm nor exclude it.
2. **Driver and CUDA — 570.211.01 + 12.9 vs 590.48.01 + 13.1.** Launch and stream-management cost differ across major CUDA versions, and ~4-7 us per operation is the right order of magnitude.
3. **Kernel — uniform 6.8 vs a heterogeneous 4.18 / 6.12 pair.** The weakest of the three: node5500 (4.18) and node5502 (6.12) show the same small-message times, so the kernel does not separate them.
4. **Ordinary variation.** Each row is one sweep of 20 iterations and the Rocky nodes vary among themselves by 5.8-15.1% at 1 MiB. The 9-of-9 sign consistency argues against pure noise, but a repeat sweep would settle it cheaply.

**Caveat.** The Rocky runs are from 2026-08-06 and the Ubuntu runs from 2026-08-10; nothing here depends on a same-day comparison, since the effects are either ties or stable across three nodes. CPU model, governor and BIOS settings could not be read on node5500-5502, so candidate 1 remains untested.

## 5. Suggested actions

The action list derived from sections 1-4 — what to check, what to change, and in what order — is in **`../admin-nccl-notes.md`**. Its section 1 holds the general items; section 2 holds those specific to the single-node case.
