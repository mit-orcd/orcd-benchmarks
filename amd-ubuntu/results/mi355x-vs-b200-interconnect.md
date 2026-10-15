# AMD MI355X vs NVIDIA B200: interconnect (RCCL vs NCCL)

Every ratio is **MI355X / B200**: above 1 means MI355X is faster (higher bandwidth or lower latency). MI355X = RCCL on ROCm 7.14, node6100 + node6101; B200 = NCCL on the B200 nodes.

- [Quick summary](#quick-summary)
- [1. Intra-node (8 GPUs in one node)](#1-intra-node-8-gpus-in-one-node): 1.1 on paper, 1.2 measured bandwidth, 1.3 measured latency
- [2. Inter-node (2 nodes, 16 GPUs)](#2-inter-node-2-nodes-16-gpus): 2.1 on paper, 2.2 measured bandwidth, 2.3 measured latency
- [Appendix A. Pollara: RoCEv2 vs Ultra Ethernet](#appendix-a-pollara-rocev2-vs-ultra-ethernet)

## Quick summary

Which is faster and by how much. Here N× faster is the faster side's speed over the slower side's; the sections below give the same results as MI355X / B200 ratios.

| | Faster | By how much |
|---|---|---|
| **Intra-node bandwidth** (8 GPUs, large messages), §1.2 | **B200** (NVLink 5 + NVSwitch) | • B200 1.7–2.1× faster on collectives<br>• B200 11× faster for one GPU pair (sendrecv) |
| **Intra-node latency** (8 GPUs, 8 B–1 MB), §1.3 | **MI355X** (Infinity Fabric / XGMI) | • MI355X 1.1–5× faster in all_reduce, all_gather, reduce_scatter, alltoall<br>• B200 faster only in broadcast from 64 KB (1.3–3.4×) and sendrecv at 1 MB (1.1×) |
| **Inter-node bandwidth** (2 nodes, large messages), §2.2 | **Equal** (Pollara vs InfiniBand) | • Within 3%<br>• sendrecv equal with `NCCL_NET=IB`<br>• MI355X 1.6× faster in all_reduce and 1.8× in alltoall, where NCCL is low |
| **Inter-node latency** (2 nodes, small messages), §2.3 | **B200** (InfiniBand) | • Network alone: InfiniBand 1.8–5× faster<br>• 1 GPU per node: B200 1.2–1.5× faster<br>• 8 GPUs per node:<br>&nbsp;&nbsp;◦ MI355X 1.1–1.2× faster in all_reduce (B200 1.06× faster at 64 KB), thanks to its faster step inside the node<br>&nbsp;&nbsp;◦ B200 1.1–1.5× faster in sendrecv and 1.1–2.7× in all_gather, reduce_scatter, broadcast, alltoall up to 64 KB<br>&nbsp;&nbsp;◦ MI355X 0.9–1.3× faster in those at 1 MB, also with B200's best fixed NCCL setting (default: 1.2–1.7×) |

On paper: links inside one node 537.6 vs 900 GB/s per GPU (0.60x, §1.1); network 400 vs 400 GB/s per node (1.00x, §2.1); neither vendor publishes latency figures.

## 1. Intra-node (8 GPUs in one node)

### 1.1 On paper: AMD Infinity Fabric (XGMI) vs NVIDIA NVLink 5

> **On paper, the B200 node is better than the MI355X node for GPU-to-GPU traffic inside one node: 900 vs 537.6 GB/s per GPU (1.67×), and any two B200 GPUs can use all of it, while two MI355X GPUs share one 76.8 GB/s link.**

Vendor specs ("on paper") for the links between the 8 GPUs inside one node. Measured numbers are in §1.2 (bandwidth) and §1.3 (latency).

| | MI355X node | B200 node (HGX B200) |
|---|---|---|
| Link | AMD **Infinity Fabric (XGMI)** | NVIDIA **NVLink 5** |
| Links per GPU | 7 | 18 |
| Speed per link, per direction | 76.8 GB/s | 50 GB/s |
| Per GPU, per direction | **537.6 GB/s** (7 × 76.8) | **900 GB/s** (18 × 50) |
| Per GPU, both directions | 1,075 GB/s | 1,800 GB/s |
| Topology | **full mesh**: each GPU has one direct link to each of the other 7, no switch | **switched**: every GPU connects to NVSwitch chips |
| One GPU pair can use | one link: **76.8 GB/s** | all of it: **900 GB/s** |
| Reduction inside the fabric | none | NVLink SHARP in the NVSwitch |
| GPUs in one link domain | 8 | 8 |
| Measured, one pair (sendrecv) | 60.4 GB/s = 79% of one link | 668.6 GB/s = 74% of 900 |
| Measured, all_reduce, 8 GPUs | 396.9 GB/s = 74% of 537.6 | 839.4 GB/s = 93% of 900 |
| Latency on paper | not published by AMD | not published by NVIDIA |
| Measured latency, sendrecv 8 B, 8 GPUs (§1.3) | 6.2 µs | 9.0 µs |
| Measured latency, all_reduce 1 KB, 8 GPUs (§1.3) | 5.0 µs | 25.2 µs |

**Same on paper:** 8 GPUs in one link domain, GPU-to-GPU traffic never leaves the node. **Different:** B200 has 1.67× the link bandwidth per GPU, any pair of B200 GPUs can use all of it (MI355X: one link per pair, 1/7 of the total), and only B200 can reduce data inside the fabric. This is why MI355X is 0.47–0.59× on collectives and 0.09× on sendrecv in one node. Neither vendor publishes a latency figure for these links; for small messages the measured time is mostly software and synchronization, not the link (§1.3).

### 1.2 Measured bandwidth (large messages)

busbw (GB/s) at the largest message size, **8 GPUs in one node on both systems** (one rank per GPU, the GPUs talk over XGMI / NVLink only). MI355X = RCCL on ROCm 7.14, mean of node6100 and node6101 (max 8 GB; alltoall 4 GB). B200 = mean of 10 nodes (max 16 GiB).

| Collective | MI355X (RCCL) | MI355X on paper | MI355X % of paper | B200 (NCCL) | B200 on paper | B200 % of paper | MI355X / B200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_reduce | 396.9 | 537.6 | 74% | 839.4 | 900 | 93% | **0.47x** |
| all_gather | 389.4 | 537.6 | 72% | 681.6 | 900 | 76% | **0.57x** |
| reduce_scatter | 388.7 | 537.6 | 72% | 691.6 | 900 | 77% | **0.56x** |
| broadcast | 389.4 | 537.6 | 72% | 682.1 | 900 | 76% | **0.57x** |
| reduce | 325.3 | 537.6 | 61% | 687.5 | 900 | 76% | **0.47x** |
| gather | 423.8 | 537.6 | 79% | 718.0 | 900 | 80% | **0.59x** |
| scatter | 395.9 | 537.6 | 74% | 731.2 | 900 | 81% | **0.54x** |
| alltoall | 347.6 | 537.6 | 65% | 658.1 | 900 | 73% | **0.53x** |
| sendrecv | 60.4 | 76.8 | 79% | 668.6 | 900 | 74% | **0.09x** |

On paper = vendor peak per GPU, one direction, GB/s:
- **MI355X**: 7 Infinity Fabric (XGMI) links × 76.8 GB/s = **537.6 GB/s** per GPU; **sendrecv** uses one GPU pair = one link = **76.8 GB/s**.
- **B200**: NVLink 5 through NVSwitch = **900 GB/s** per GPU, for every collective (a pair can use all of it, so also for sendrecv).
- **Note on all_reduce (expected):** B200 reaches 93% of its link peak, above its ≈76% on the other collectives, because **NVLink SHARP** (the NVSwitch adds the data up inside the switch) can improve all_reduce on B200 nodes; NCCL uses it by default when available (not checked in the B200 logs). MI355X has no switch and no such feature, so its 74% is normal for it, and the ratio (0.47x) falls a little below the 0.60x link ratio on paper.

**Apple-to-apple: close, not exact.**
- Same test design: one process driving 8 GPUs, 20 iterations after 5 warm-up, busbw at the largest size.
- Largest message differs (8 GB vs 16 GiB). Both curves are flat at that point, so the effect is small.
- The ratio reflects the hardware: MI355X connects the GPUs in a full mesh of XGMI links (537.6 GB/s per GPU in total, 76.8 GB/s per pair on paper). B200 connects all GPUs through NVSwitch (900 GB/s per GPU, any pair can use all of it). That is why sendrecv, which uses one pair of GPUs, is 0.09x.
- MI355X with 5–7 GPUs drops to ≈47 GB/s (see `ubuntu/rocm.md`); B200 has no such dip.

> [!NOTE]
> Inside one node, NVIDIA NVLink with NVSwitch is faster: AMD Infinity Fabric reaches only 0.47–0.59x of its collective bandwidth.

<!-- INTRA-LAT-BEGIN (auto: rccl-tests/compare_lat_1node.py) -->
### 1.3 Measured latency (8 GPUs, small messages)

rccl-tests (MI355X, ROCm 7.14) / nccl-tests (B200), 8 B–4 MB, 1000 iterations + 100 warm-up, one rank per GPU, NUMA-bound, each run started only after all GPUs read 0 % use. Time per call in µs, lower is better; speed = B200 time ÷ MI355X time (>1: MI355X faster). Default = median of 3 default runs per node (MI355X: node6100, node6101; B200: node5700-c1, node5801-c1).

> **Mixed** with default settings, 8 B–1 MB. sendrecv: **mixed** (0.91–1.46x); all_reduce: **MI355X faster** (1.29–5.06x); all_gather: **MI355X faster** (1.56–2.05x); reduce_scatter: **MI355X faster** (2.07–2.62x); broadcast: **mixed** (0.30–1.28x); alltoall: **MI355X faster** (1.06–1.29x).

| collective | bytes | MI355X (XGMI) | B200 (NVLink) | MI355X / B200 speed | best fixed setting¹, MI355X (µs) | best fixed setting¹, B200 (µs) |
|---|---:|---:|---:|---:|---:|---:|
| sendrecv | 8 B | 6.2 | 9.0 | 1.46x | 6.2 <sup>b</sup> | 8.3 <sup>a</sup> |
| sendrecv | 1 KB | 6.5 | 8.9 | 1.38x | 6.4 <sup>a</sup> | 8.4 <sup>a</sup> |
| sendrecv | 64 KB | 13.4 | 15.3 | 1.14x | 13.3 <sup>a</sup> | 14.1 <sup>a</sup> |
| sendrecv | 1 MB | 29.5 | 27.0 | 0.91x | 29.5 <sup>a</sup> | 25.1 <sup>a</sup> |
| all_reduce | 8 B | 14.4 | 18.5 | 1.29x | 12.4 <sup>a</sup> | 17.0 <sup>d</sup> |
| all_reduce | 1 KB | 5.0 | 25.2 | 5.06x | 4.9 <sup>b</sup> | 23.4 <sup>d</sup> |
| all_reduce | 64 KB | 6.5 | 26.8 | 4.13x | 6.5 <sup>a</sup> | 25.2 <sup>d</sup> |
| all_reduce | 1 MB | 17.4 | 28.1 | 1.62x | 17.4 <sup>c</sup> | 26.1 <sup>d</sup> |
| all_gather | 1 KB | 7.8 | 15.9 | 2.05x | 7.7 <sup>c</sup> | 15.9 <sup>c</sup> |
| all_gather | 64 KB | 8.9 | 17.6 | 1.96x | 8.8 <sup>a</sup> | 17.5 <sup>c</sup> |
| all_gather | 1 MB | 11.7 | 18.3 | 1.56x | 11.7 <sup>b</sup> | 18.2 <sup>c</sup> |
| reduce_scatter | 1 KB | 6.1 | 15.9 | 2.62x | 5.9 <sup>c</sup> | 15.6 <sup>c</sup> |
| reduce_scatter | 64 KB | 7.1 | 17.1 | 2.42x | 7.0 <sup>c</sup> | 16.9 <sup>a</sup> |
| reduce_scatter | 1 MB | 8.7 | 18.0 | 2.07x | 8.6 <sup>b</sup> | 17.8 <sup>c</sup> |
| broadcast | 8 B | 5.7 | 6.9 | 1.21x | 5.6 <sup>c</sup> | 6.3 <sup>a</sup> |
| broadcast | 1 KB | 5.8 | 7.4 | 1.28x | 5.7 <sup>a</sup> | 7.1 <sup>c</sup> |
| broadcast | 64 KB | 15.0 | 12.0 | 0.80x | 15.0 <sup>a</sup> | 11.4 <sup>c</sup> |
| broadcast | 1 MB | 41.7 | 12.3 | 0.30x | 19.6 <sup>c</sup> | 11.8 <sup>c</sup> |
| alltoall | 1 KB | 7.6 | 9.8 | 1.29x | 7.5 <sup>b</sup> | 9.1 <sup>d</sup> |
| alltoall | 64 KB | 10.4 | 11.1 | 1.06x | 10.2 <sup>a</sup> | 10.4 <sup>d</sup> |
| alltoall | 1 MB | 14.0 | 17.7 | 1.26x | 13.5 <sup>c</sup> | 16.4 <sup>d</sup> |

¹ Fastest of the fixed settings: <sup>a</sup> `NCCL_ALGO=Ring NCCL_PROTO=LL`, <sup>b</sup> `NCCL_ALGO=Tree NCCL_PROTO=LL`, <sup>c</sup> `NCCL_PROTO=LL`, <sup>d</sup> `NCCL_NVLS_ENABLE=0`.

**Run-to-run spread of the default runs** (min–max over all runs and nodes, µs):

| collective | bytes | MI355X | B200 |
|---|---:|---:|---:|
| sendrecv | 8 B | 6.0–6.5 | 8.8–9.9 |
| sendrecv | 1 KB | 6.4–6.6 | 8.8–9.8 |
| all_reduce | 8 B | 13.2–14.7 | 17.7–19.7 |
| all_reduce | 1 KB | 4.9–5.1 | 24.3–27.2 |

All nodes, settings and sizes: [`latency-1node-mi355x-vs-b200.md`](latency-1node-mi355x-vs-b200.md).

Sources: `../logs/node6100/rocm7.14/rccl/lat_1node_20261014_222719`, `../logs/node6101/rocm7.14/rccl/lat_1node_20261014_222717`, `../../b200-nodes/out-nccl-lat-1node/25863885`, `../../b200-nodes/out-nccl-lat-1node/25863886`. Updated 2026-10-15.
<!-- INTRA-LAT-END -->

**What this shows.**
- **For small messages, MI355X is faster inside one node, not B200.** NVLink's higher bandwidth (§1.2) helps large messages; below about 64 KB the time is mostly software and synchronization, and RCCL on XGMI is quicker there. B200 is faster only in broadcast from 64 KB and sendrecv at 1 MB.
- **The earlier 1-node numbers were real, not a busy-GPU effect.** All new runs started with every GPU at 0 % use and repeat the earlier values within a few percent. B200 gives the same result on three different nodes (node5600-c1 earlier, node5700-c1 and node5801-c1 now), and MI355X on both node6100 and node6101.
- **B200 all_reduce stays at about 17–28 µs from 8 B to 1 MB with every setting tried.** Turning NVLS off, or forcing Ring or Tree with the LL protocol, saves at most about 2 µs, so NCCL's choice of algorithm is not the cause. The cause is not found yet.
- **MI355X all_reduce at 8 B (about 14 µs) is slower than at 64 B–1 KB (5 µs) in every run.** 8 bytes is only 2 values for 8 GPUs, so RCCL likely takes a different path for such a tiny message (our guess, not checked in RCCL's code). From 64 B up the times are normal.
- **MI355X broadcast at 1 MB is slow with RCCL's default (41.7 µs); forcing the LL protocol halves it (about 20 µs on both nodes, alone or with Ring; Tree + LL does not help).** Here RCCL's default choice costs time.

## 2. Inter-node (2 nodes, 16 GPUs)

### 2.1 On paper: AMD Pollara (Ethernet) vs NVIDIA InfiniBand

Vendor specs ("on paper"), plus what is set on these clusters. Measured numbers are in §2.2 (bandwidth), §2.3 (latency) and `ubuntu/net.md`.

| | MI355X nodes (node6100/6101) | B200 nodes |
|---|---|---|
| NIC | AMD Pensando **Pollara 400** AI NIC (`ionic` driver) | NVIDIA **ConnectX-7** (MT4129, `mlx5` driver) |
| Network type | **Ethernet**. Pollara is sold as Ultra Ethernet (UEC 1.0) ready; **here it runs RoCEv2** (RDMA over Converged Ethernet), not the UEC transport (see [Appendix A](#appendix-a-pollara-rocev2-vs-ultra-ethernet)) | **InfiniBand NDR** |
| Speed per NIC | 400 Gb/s = 50 GB/s per direction | 400 Gb/s = 50 GB/s per direction |
| NICs per node | 8, one per GPU (rails 0–3 on CPU socket 0, 4–7 on socket 1) | 8, one per GPU |
| Per node | 3.2 Tb/s = 400 GB/s per direction | 3.2 Tb/s = 400 GB/s per direction |
| Host link | PCIe Gen5 x16 | PCIe Gen5 x16 |
| GPU-direct RDMA | yes | yes |
| Loss handling | Ethernet, lossy by default; UEC-style multipath (packet spraying), selective retransmit and programmable congestion control (P4-programmable NIC) | lossless link layer (credit-based flow control); adaptive routing in the switch |
| In-network reduction | none | SHARP v3 in the Quantum-2 switches (tested in `../b200-nodes/out-nccl-2node-sharp/`) |
| Switch | Ethernet switch (model not recorded) | NVIDIA Quantum-2 NDR |
| MTU (maximum transmission unit: largest packet size on the link, bytes) | 4096 (RoCE) | 4096 (IB) |
| Measured, one rail | `ib_write_bw` 392 Gb/s = 98% of line rate | nccl sendrecv 48.8 GB/s = 98% of one rail |
| Latency on paper (NIC, switch) | not published by AMD | not published by NVIDIA ("ultra-low latency") |
| Measured latency, one rail, no GPU (`ib_write_lat` 8 B, one way, §2.3) | 5.21 µs | 1.33 µs |

**Same on paper:** line rate per NIC and per node, NICs per GPU, PCIe Gen5. **Different:** Ethernet (RoCEv2) vs InfiniBand, and only the B200 network can reduce data inside the switch (SHARP). The UEC features of Pollara are not used in these runs. Neither vendor publishes a latency figure; measured, one rail with no GPU, Pollara is about 4× slower than InfiniBand (5.2 vs 1.3 µs one way).

### 2.2 Measured bandwidth (large messages)

**Compared: MI355X (RCCL, ROCm 7.14, node6100 + node6101) vs B200 (NCCL, mean of the 4 healthy node pairs), 2 nodes × 8 GPUs, one rank per GPU.** busbw (GB/s) at the largest message size (16 GiB; MI355X alltoall 4 GB). Both have 8 × 400G rails per node = 400 GB/s (§2.1). RCCL is AMD's version of NCCL and reads the same `NCCL_*` setting names, so the `NCCL_*` variables below are RCCL settings on MI355X.

| Collective | MI355X (RCCL) | MI355X ceiling | MI355X % of ceiling | B200 (NCCL) | B200 ceiling | B200 % of ceiling | MI355X / B200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_reduce | 372.6 | 400 | 93% | $\color{red}{\textsf{233.1}}$ | 400 | $\color{red}{\textsf{58\\%}}$ | $\color{red}{\textbf{1.60x}}$ |
| all_gather | 372.3 | 400 | 93% | 375.4 | 400 | 94% | 0.99x |
| reduce_scatter | 371.6 | 400 | 93% | 382.4 | 400 | 96% | 0.97x |
| broadcast | 364.6 | 400 | 91% | 359.1 | 400 | 90% | 1.02x |
| alltoall⁴ | 89.6 | **93.75**¹ | 96% | $\color{red}{\textsf{48.9}}$ | **93.75**¹ | $\color{red}{\textsf{52\\%}}$ | $\color{red}{\textbf{1.83x}}$ |
| sendrecv³ ⁴ | 49.4 | **50**² | 99% | 48.8 | **50**² | 98% | 1.01x |

Ceiling = what the network can carry for that pattern, per node and direction. **400 GB/s** for the ring/tree collectives: up to 8 rings, one per rail, together use the whole network of the node. sendrecv and alltoall use no ring, so their ceiling is what one GPU can send over its own rail:
- ¹ **alltoall, 93.75 GB/s:** each GPU sends half of its data straight to the other node over its own rail (50 GB/s), so the call runs at up to 100 GB/s; in busbw units that is 100 × 15/16. It is lower than the 400 GB/s of the ring collectives because a ring spreads each GPU's data over all 8 rails, while in alltoall each rail carries half of one GPU's data, about 4× more. It is an upper bound: protocol overhead makes the real ceiling about 5% lower.
- ² **sendrecv, 50 GB/s:** each GPU pair that crosses the nodes uses one rail, and the slowest pair sets the busbw.
- ³ MI355X sendrecv result: `NCCL_NET=IB` plus `NCCL_NCHANNELS_PER_NET_PEER=4` (4 channels per network peer).
- ⁴ **Fixed results:** ROCm 7.14 with `NCCL_NET=IB`. With RCCL 7.14's defaults these two are lower: alltoall 77.1 GB/s (82%, 1.58x of B200), sendrecv 32.2 GB/s (64%; 39.9 = 80% with 4 channels). The ring collectives above use the defaults (not rerun with `NCCL_NET=IB`).

**How it is fixed:** ROCm 7.14's RCCL uses a new network transport (`IB-CAST`) by default, which slows point-to-point traffic between nodes. `NCCL_NET=IB` switches back to the standard transport; for sendrecv, `NCCL_NCHANNELS_PER_NET_PEER=4` also adds channels (set it per job: it lowers alltoall by ≈30%). With the defaults MI355X sendrecv reaches only 64%; with both settings 99% (1.01x of B200). Found step by step (2026-10-08; `ubuntu/sendrecv-check.md`, logs in `logs/node6100/rocm7.14/rccl/`):
  1. **System and network are fine:** GPUDirect RDMA is on for all 8 rails, the ring collectives use the same GPU-to-NIC path at 91–93%, and raw RDMA (`ib_write_bw`) reaches 98% of a rail.
  2. **Too few channels per network peer:** 4 channels lift sendrecv from 64% to 80%.
  3. **ROCm 7.14's new `IB-CAST` transport:** switching back with `NCCL_NET=IB` lifts it to 99%, the same as ROCm 7.2.4, so the gap is in RCCL 7.14, not the hardware or network.
- Inside one node (§1.2) alltoall needs no special ceiling: each GPU has its own links to every other GPU (B200: NVSwitch).

> [!NOTE]
> Between two nodes, AMD Pollara (Ethernet) performs the same as NVIDIA InfiniBand: all_gather, reduce_scatter and broadcast are equal (0.97–1.02x), both near the full network. MI355X is faster in all_reduce (1.60x) and alltoall (1.83x), because NCCL is low there; B200 all_reduce could improve with SHARP (data reduced inside the InfiniBand switch), which was not used here. sendrecv is equal (1.01x) once RCCL is set up as above.

**Apple-to-apple: nearly.** Same test, 16 ranks, same message range, same network speed. Differences: RoCEv2 (AMD Pollara) vs InfiniBand (NVIDIA), RCCL vs NCCL. reduce, gather and scatter were not run on MI355X across 2 nodes.

**Results far from paper** (also listed in the Quick summary of [`mi355x-vs-b200.md`](mi355x-vs-b200.md)):

- **B200 all_reduce reaches only 58% (1.60x vs 1.00x on paper):** other NCCL collectives reach 90–96% on the same network, so **the likely cause is NCCL's all_reduce algorithm, not the network** (the algorithm NCCL picked was not logged). **SHARP should help:** the switches add the data up, so each GPU sends it once instead of twice (2.2x on a reference cluster). It could not be tested: the fabric has no SHARP Aggregation Manager (`sharp_am`) running, a switch-side setting we can turn on with admin access (`../b200-nodes/sharp.md`).
- **B200 alltoall reaches only 52% (1.83x vs 1.00x on paper):** NCCL's alltoall is known to be slow here: like RCCL, it sends a separate piece directly to each of the 15 other GPUs, but reaches only half of the ceiling. **Likely the same cause as MI355X sendrecv: too few channels per network peer.** Not tested on B200; `NCCL_NCHANNELS_PER_NET_PEER=4` would be the first setting to try.

**ROCm 7.14 vs 7.2.4 on our nodes** (same nodes and tests, only ROCm differs): the ring collectives are the same (1–2% lower on 7.14, within noise). Point-to-point is slower on 7.14 because of the new transport above: alltoall 0.86x (77.1 vs 89.6 GB/s), sendrecv with 4 channels 0.85x (39.9 vs 46.9). Only default sendrecv is faster on 7.14 (1.12x: 32.2 vs 28.6 GB/s).

<!-- LATENCY-BEGIN (auto: rccl-tests/compare_lat.py) -->
### 2.3 Measured latency (small messages)

MI355X ROCm 7.14 (RCCL, AMD Pollara RoCEv2) vs B200 (NCCL, NVIDIA InfiniBand). Time per call in µs, lower is better; speed = B200 time ÷ MI355X time (>1: MI355X faster). MI355X = the faster of RCCL's default and `NCCL_NET=IB` (both in `ubuntu/latency.md`).

#### Raw network, no GPU: Pollara vs InfiniBand

> **InfiniBand is faster.** Pollara is 0.20–0.56x the speed of InfiniBand (perftest one-way latency, median of the 8 rails).

| test | bytes | AMD Pollara (RoCEv2) | NVIDIA InfiniBand (ConnectX-7) | Pollara / InfiniBand speed |
|---|---:|---:|---:|---:|
| ib_write_lat | 8 | 5.21 | 1.33 | 0.26x |
| ib_write_lat | 4K | 6.34 | 2.33 | 0.37x |
| ib_write_lat | 64K | 7.97 | 4.48 | 0.56x |
| ib_send_lat | 8 | 6.72 | 1.36 | 0.20x |
| ib_send_lat | 4K | 7.91 | 2.33 | 0.30x |
| ib_send_lat | 64K | 9.28 | 4.46 | 0.48x |

#### Main result: 2 nodes × 1 GPU and 2 nodes × 8 GPUs, same tests

**2 nodes × 1 GPU** (one GPU per node, so only the network is between them):

> **B200 (InfiniBand) is faster.** sendrecv: **B200 faster** at every size (0.75–0.83x); all_reduce: **B200 faster** at every size (0.65–0.85x).

| collective | bytes | MI355X (Pollara) | B200 (InfiniBand) | MI355X / B200 speed |
|---|---:|---:|---:|---:|
| sendrecv | 8 | 27.4 | 21.2 | 0.77x |
| sendrecv | 1K | 27.8 | 21.0 | 0.75x |
| sendrecv | 64K | 37.0 | 28.8 | 0.78x |
| sendrecv | 1M | 63.1 | 52.2 | 0.83x |
| all_reduce | 8 | 20.2 | 14.5 | 0.72x |
| all_reduce | 1K | 21.0 | 14.6 | 0.69x |
| all_reduce | 64K | 36.9 | 24.0 | 0.65x |
| all_reduce | 1M | 67.7 | 57.6 | 0.85x |

**2 nodes × 8 GPUs, network part only** (network part = 2-node time − 1-node time, same test and size; this removes the step among the 8 GPUs inside each node):

> **InfiniBand is faster** in 7 of 8 rows: Pollara is 0.29–0.91x the speed of InfiniBand in the network part. Exception: all_reduce 1 MB (1.10x).

| collective | bytes | network part, Pollara | network part, InfiniBand | Pollara / InfiniBand speed | whole call, MI355X / B200 speed |
|---|---:|---:|---:|---:|---:|
| sendrecv | 8 | 14.5 | 6.6 | 0.45x | 0.75x |
| sendrecv | 1K | 14.1 | 6.5 | 0.46x | 0.76x |
| sendrecv | 64K | 9.9 | 5.2 | 0.53x | 0.88x |
| sendrecv | 1M | 19.6 | 5.6 | 0.29x | 0.67x |
| all_reduce | 8 | 14.2 | 12.9 | 0.91x | 1.12x |
| all_reduce | 1K | 25.7 | 8.6 | 0.34x | 1.10x |
| all_reduce | 64K | 35.5 | 12.3 | 0.35x | 0.94x |
| all_reduce | 1M | 49.2 | 54.1 | 1.10x | 1.24x |

**Only in 2 nodes × 8 GPUs** (other collectives; speed only, >1: MI355X faster):

| collective | 256 B | 64 KB | 1 MB |
|---|---:|---:|---:|
| all_gather | 0.81x | 0.89x | 1.52x |
| reduce_scatter | 0.82x | 0.78x | 1.71x |
| broadcast | 0.81x | 0.37x | 1.18x |
| alltoall | 0.57x | 0.83x | 1.34x |

Sources: `logs/node6100/rocm7.14/rccl/lat_2node_20261013_014214`, `../../b200-nodes/out-nccl-lat/25690497`. Updated 2026-10-15.
<!-- LATENCY-END -->

**Why the two layouts differ.**
- **2 nodes × 1 GPU:** only the network is between the GPUs, and Pollara has about 4× the latency of InfiniBand (5.2 vs 1.3 µs one way), so B200 is faster in every test. The gap (0.65–0.85x) looks smaller than in the raw network (0.20–0.56x) only because both systems add about 14–20 µs of GPU and library time; Pollara still costs 4–6 µs more per crossing.
- **2 nodes × 8 GPUs:** each call also includes the step among the 8 GPUs inside each node (XGMI / NVLink). For all_reduce that step is faster on MI355X (8 B: 14.3 vs 18.4 µs), which outweighs the slower network, so MI355X is faster at most sizes. In sendrecv the slowest pair crosses the network, so B200 stays faster.
<!-- B200-TUNE-BEGIN (auto: rccl-tests/b200_lat_tune.py) -->
- **1 MB, B200 rerun with fixed `NCCL_ALGO` / `NCCL_PROTO` (job 25990596):** **Partly confirmed.** A fixed setting makes B200 faster at 1 MB in 3 of 4 collectives, but MI355X stays faster in 3 of them (1.16–1.69x with NCCL's default, 0.88–1.33x with B200's best fixed setting). This job's default reproduces the earlier run within 1%.

  | collective | bytes | B200 default, earlier run | B200 default, this job | B200 best fixed | best setting | MI355X | MI355X / B200 speed, default | MI355X / B200 speed, B200 best fixed |
  |---|---:|---:|---:|---:|---|---:|---:|---:|
  | all_gather | 64 KB | 34.0 | 36.7 | 36.7 | none faster than default | 38.2 | 0.96x | 0.96x |
  | all_gather | 1 MB | 83.1 | 84.2 | 61.9 | NCCL_ALGO=Ring NCCL_PROTO=LL | 54.5 | 1.54x | 1.13x |
  | all_gather | 4 MB | 81.5 | 83.2 | 83.1 | none faster than default | 73.0 | 1.14x | 1.14x |
  | reduce_scatter | 64 KB | 33.6 | 35.7 | 35.5 | none faster than default | 43.3 | 0.82x | 0.82x |
  | reduce_scatter | 1 MB | 93.9 | 93.2 | 61.2 | NCCL_PROTO=LL | 55.0 | 1.69x | 1.11x |
  | reduce_scatter | 4 MB | 89.3 | 90.8 | 90.9 | none faster than default | 72.4 | 1.25x | 1.26x |
  | broadcast | 64 KB | 16.9 | 19.8 | 19.7 | none faster than default | 45.5 | 0.44x | 0.43x |
  | broadcast | 1 MB | 70.9 | 70.0 | 53.1 | NCCL_ALGO=Ring NCCL_PROTO=LL128 | 60.1 | 1.16x | 0.88x |
  | broadcast | 4 MB | 78.7 | 80.7 | 80.0 | none faster than default | 78.5 | 1.03x | 1.02x |
  | alltoall | 64 KB | 30.5 | 31.3 | 31.1 | none faster than default | 36.6 | 0.85x | 0.85x |
  | alltoall | 1 MB | 62.3 | 63.1 | 62.1 | none faster than default | 46.6 | 1.35x | 1.33x |
  | alltoall | 4 MB | 133.2 | 142.8 | 134.6 | NCCL_ALGO=Ring NCCL_PROTO=Simple | 103.2 | 1.38x | 1.30x |
  | all_reduce | 64 KB | 39.4 | 40.9 | 48.9 | none faster than default | 41.9 | 0.98x | 1.17x |
  | all_reduce | 1 MB | 82.3 | 84.6 | 78.8 | NCCL_ALGO=Tree NCCL_PROTO=LL | 66.5 | 1.27x | 1.18x |
  | all_reduce | 4 MB | 98.9 | 100.2 | 96.0 | NCCL_PROTO=Simple | 91.0 | 1.10x | 1.05x |
  | sendrecv | 64 KB | 20.4 | 21.2 | 20.2 | NCCL_ALGO=Tree NCCL_PROTO=LL | 23.2 | 0.91x | 0.87x |
  | sendrecv | 1 MB | 32.7 | 35.9 | 34.4 | NCCL_ALGO=Tree | 49.1 | 0.73x | 0.70x |
  | sendrecv | 4 MB | 100.1 | 99.3 | 95.7 | NCCL_ALGO=Tree | 118.5 | 0.84x | 0.81x |

  Time per call, µs. All settings: `../../b200-nodes/out-nccl-lat-tune/25990596/summary.md`.
<!-- B200-TUNE-END -->

## Appendix A. Pollara: RoCEv2 vs Ultra Ethernet

**RoCEv2 and Ultra Ethernet are different protocols.** RoCEv2 is the InfiniBand transport carried in UDP/IP packets over Ethernet (an IBTA standard from 2014). Ultra Ethernet Transport (UET, UEC 1.0) is a new transport written from scratch for AI and HPC; it also runs over UDP/IP Ethernet, but it does not reuse the InfiniBand transport. A RoCEv2 NIC and a UET NIC cannot talk to each other with RDMA.

| | RoCEv2 | Ultra Ethernet Transport (UEC 1.0) |
|---|---|---|
| Origin | InfiniBand transport over Ethernet (IBTA) | new design (Ultra Ethernet Consortium, 2025) |
| Software interface | RDMA verbs (queue pairs) | libfabric |
| Connections | one queue pair per peer, state kept for each | short-lived, set up on demand, so it scales to very many peers |
| Packet order | must arrive in order | may arrive out of order (ordered delivery is optional) |
| Paths | one path per connection | packets sprayed over many paths |
| Loss recovery | go-back-N | selective retransmission |
| Congestion control | DCQCN, needs ECN and PFC (lossless) tuned in the switches | new sender- and receiver-based schemes, works on a lossy network |
| Security | none in the protocol | optional encryption built in |

**What "UEC ready" means: Pollara 400 sits in between.** It speaks standard RoCEv2, so it works with other NICs and switches, and its P4-programmable packet pipeline adds some UET-style features in firmware (listed below). It does not run the full UET protocol.

**What runs on node6100/6101.** Standard RoCEv2. Checked on node6100: all 8 NICs (`ionic_0`–`ionic_7`, firmware 1.117.5-a-196) use RoCE v2 GIDs (`NCCL_IB_GID_INDEX=1`), and RCCL opens them through the normal verbs path (`NET/IB/<n>/GDRDMA`, or `NET/IB-CAST` with ROCm 7.14's default). Nothing in our runs turns on the UEC-style features, so every result in this file is plain RoCEv2.

**Pollara features beyond plain RoCEv2, and how to turn them on.** The NIC hardware has them (AMD's description, not tested here):

| Feature | What it changes vs plain RoCEv2 |
|---|---|
| Packet spray | one connection's packets use many paths instead of one |
| Out-of-order placement | every packet carries its memory address, so packets can be written to GPU memory in any order |
| Selective retransmission | only lost packets are resent, not everything after them |
| Path-aware congestion avoidance | traffic is steered away from busy paths |

AMD's own figure: about 25% more all_reduce bandwidth than plain RoCEv2 (vendor test).

**Would they help on these two nodes? Probably only a little.** With 2 nodes, 8 rails and no other traffic there is little to spray around and almost no packet loss, and plain RoCEv2 already reaches 91–99% of the line rate (§2.2). The latency gap (5.2 vs 1.3 µs one way, §2.3) is not expected to change much either. The features mainly help large fabrics with several switch tiers, many jobs and congestion, so they matter more if more Pollara nodes are added.

Turning them on needs root and switch access, so **we do it with our admin accounts, using software and settings from AMD** (`nicctl` needs root):

| Needed | Who | Now on node6100/6101 |
|---|---|---|
| Firmware and driver with the features | AMD provides, we install | firmware 1.117.5-a-196, `ionic` driver 26.08.17.001 |
| NIC settings (transport mode, congestion control) | we, with `nicctl` (root) | defaults |
| Switch settings (ECN, QoS marking, per-packet load balancing) | we, with the network team | not checked |
| RCCL network plugin for Pensando NICs (AMD's ANP plugin) | AMD provides, we install | not installed; RCCL uses its built-in `NET/IB` |

**Our next steps:**
1. Ask AMD which firmware, driver, `nicctl` and switch settings enable these features on Pollara 400, and whether the ANP RCCL plugin is needed (the plugin name and `nicctl` options above are from memory; AMD should confirm them).
2. Install and configure them on node6100 and node6101 during a maintenance window.
3. Rerun §2.2 and §2.3 to measure the difference.

**Next step by AMD: Vulcano 800G AI NIC** (successor, PCIe Gen6, 800 Gb/s), announced for 2026 with the MI400 "Helios" rack. AMD says it is UEC 1.0 compliant (the full UEC transport, not only "ready") and will support AMD's Multipath Reliable Connection (MRC) transport. Shipping status not confirmed as of 2026-10.

Sources: [AMD Pollara 400 product brief](https://www.amd.com/content/dam/amd/en/documents/pensando-technical-docs/product-briefs/pensando-pollara-400-product-brief.pdf), [AMD blog: Transforming AI networks with Pollara 400](https://www.amd.com/en/blogs/2024/transforming-ai-networks-with-amd-pensando-pollar.html), [AMD blog: Vulcano 800 AI NIC](https://www.amd.com/en/blogs/2026/amd-pensando-vulcano-800-ai-nic-scale-out-and-across.html), [ServeTheHome: Vulcano 800G NIC and UEC plans](https://www.servethehome.com/amd-vulcano-800g-nic-coming-as-amd-outlines-its-ualink-and-uec-scale-plans/).
