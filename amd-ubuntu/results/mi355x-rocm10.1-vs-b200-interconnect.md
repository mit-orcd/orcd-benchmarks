# AMD MI355X vs NVIDIA B200: interconnect (RCCL vs NCCL)

Every ratio is **MI355X / B200**: above 1 means MI355X is faster (higher bandwidth or lower latency). MI355X = RCCL 2.30.7 on ROCm 10.1, node6100 + node6101; B200 = NCCL on the B200 nodes.

- [Quick summary](#quick-summary)
- [1. Intra-node (8 GPUs in one node)](#1-intra-node-8-gpus-in-one-node): 1.1 on paper, 1.2 measured bandwidth, 1.3 measured latency
- [2. Inter-node (2 nodes, 16 GPUs)](#2-inter-node-2-nodes-16-gpus): 2.1 on paper, 2.2 measured bandwidth, 2.3 measured latency
- [Appendix A. Pollara: RoCEv2 vs Ultra Ethernet](#appendix-a-pollara-rocev2-vs-ultra-ethernet)

## Quick summary

Which is faster and by how much. Here N× faster is the faster side's speed over the slower side's; the sections below give the same results as MI355X / B200 ratios.

| | Faster | By how much |
|---|---|---|
| **Intra-node bandwidth** (8 GPUs, large messages), §1.2 | **B200** (NVLink 5 + NVSwitch) | • B200 1.7–2.1× faster on collectives<br>• B200 11× faster for one GPU pair (sendrecv) |
| **Intra-node latency** (8 GPUs, 8 B–1 MB), §1.3 | **MI355X** (Infinity Fabric) | • MI355X 1.2–5.0× faster in all_reduce, all_gather, reduce_scatter, alltoall<br>• B200 faster only in sendrecv at 1 MB (1.1×); broadcast at 1 KB, 64 KB, 1 MB (1.1–3.4×) |
| **Inter-node bandwidth** (2 nodes, large messages), §2.2 | **Equal** (Pollara vs InfiniBand) | • Within 5%<br>• sendrecv 1.01x with `NCCL_NET=IB` + `NCCL_NCHANNELS_PER_NET_PEER=4`<br>• MI355X 1.6× faster in all_reduce and 1.8× in alltoall, where NCCL is low |
| **Inter-node latency** (2 nodes, small messages), §2.3 | **B200** (InfiniBand) | • Network alone: InfiniBand 1.8–5× faster<br>• 1 GPU per node: MI355X / B200 0.55–1.05x<br>• 8 GPUs per node: all_reduce MI355X / B200 0.79–1.43x; sendrecv MI355X / B200 0.68–0.93x |

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

busbw (GB/s) at the largest message size, **8 GPUs in one node on both systems** (one rank per GPU, the GPUs talk over XGMI / NVLink only). MI355X = RCCL on ROCm 10.1, mean of node6100 and node6101 (max 8 GB; alltoall 4 GB). B200 = mean of 10 nodes (max 16 GiB).

| Collective | MI355X (RCCL) | MI355X on paper | MI355X % of paper | B200 (NCCL) | B200 on paper | B200 % of paper | MI355X / B200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_reduce | 395.3 | 537.6 | 74% | 839.4 | 900 | 93% | **0.47x** |
| all_gather | 388.8 | 537.6 | 72% | 681.6 | 900 | 76% | **0.57x** |
| reduce_scatter | 386.4 | 537.6 | 72% | 691.6 | 900 | 77% | **0.56x** |
| broadcast | 388.5 | 537.6 | 72% | 682.1 | 900 | 76% | **0.57x** |
| reduce | 329.7 | 537.6 | 61% | 687.5 | 900 | 76% | **0.48x** |
| gather | 423.4 | 537.6 | 79% | 718.0 | 900 | 80% | **0.59x** |
| scatter | 398.5 | 537.6 | 74% | 731.2 | 900 | 81% | **0.54x** |
| alltoall | 347.1 | 537.6 | 65% | 658.1 | 900 | 73% | **0.53x** |
| sendrecv | 60.2 | 76.8 | 78% | 668.6 | 900 | 74% | **0.09x** |

On paper = vendor peak per GPU, one direction, GB/s:
- **MI355X**: 7 Infinity Fabric (XGMI) links × 76.8 GB/s = **537.6 GB/s** per GPU; **sendrecv** uses one GPU pair = one link = **76.8 GB/s**.
- **B200**: NVLink 5 through NVSwitch = **900 GB/s** per GPU, for every collective (a pair can use all of it, so also for sendrecv).
- **Note on all_reduce (expected):** B200 reaches 93% of its link peak, above its ≈76% on the other collectives, because **NVLink SHARP** (the NVSwitch adds the data up inside the switch) can improve all_reduce on B200 nodes; NCCL uses it by default when available (not checked in the B200 logs). MI355X has no switch and no such in-network reduction.

**Apple-to-apple: close, not exact.**
- Same test design: one process driving 8 GPUs, 20 iterations after 5 warm-up, busbw at the largest size.
- Largest message differs (8 GB vs 16 GiB). Both curves are flat at that point, so the effect is small.
- The ratio reflects the hardware: MI355X connects the GPUs in a full mesh of XGMI links (537.6 GB/s per GPU in total, 76.8 GB/s per pair on paper). B200 connects all GPUs through NVSwitch (900 GB/s per GPU, any pair can use all of it). That is why sendrecv, which uses one pair of GPUs, is 0.09x.

> [!NOTE]
> Inside one node, NVIDIA NVLink with NVSwitch is faster: AMD Infinity Fabric reaches only 0.47–0.59x of its collective bandwidth.

<!-- INTRA-LAT-BEGIN (auto: rccl-tests/compare_lat_1node.py) -->
### 1.3 Measured latency (8 GPUs, small messages)

rccl-tests (MI355X, ROCm 10.1) / nccl-tests (B200), 8 B–4 MB, 1000 iterations + 100 warm-up, one rank per GPU, NUMA-bound, each run started only after all GPUs read 0 % use. Time per call in µs, lower is better; speed = B200 time ÷ MI355X time (>1: MI355X faster). Default = median of 3 default runs per node (MI355X: node6100, node6101; B200: node5700-c1, node5801-c1).

> **Mixed** with default settings, 8 B–1 MB. sendrecv: **mixed** (0.90–1.28x); all_reduce: **MI355X faster** (1.18–5.00x); all_gather: **MI355X faster** (2.33–3.14x); reduce_scatter: **MI355X faster** (2.05–2.55x); broadcast: **mixed** (0.29–1.13x); alltoall: **MI355X faster** (1.45–2.04x).

| collective | bytes | MI355X (XGMI) | B200 (NVLink) | MI355X / B200 speed | best fixed setting¹, MI355X (µs) | best fixed setting¹, B200 (µs) |
|---|---:|---:|---:|---:|---:|---:|
| sendrecv | 8 B | 7.0 | 9.0 | 1.28x | 6.9 <sup>c</sup> | 8.3 <sup>a</sup> |
| sendrecv | 1 KB | 8.4 | 8.9 | 1.06x | 8.3 <sup>c</sup> | 8.4 <sup>a</sup> |
| sendrecv | 64 KB | 14.0 | 15.3 | 1.09x | 13.8 <sup>c</sup> | 14.1 <sup>a</sup> |
| sendrecv | 1 MB | 30.0 | 27.0 | 0.90x | 30.0 <sup>a</sup> | 25.1 <sup>a</sup> |
| all_reduce | 8 B | 15.8 | 18.5 | 1.18x | 13.6 <sup>a</sup> | 17.0 <sup>d</sup> |
| all_reduce | 1 KB | 5.0 | 25.2 | 5.00x | 4.9 <sup>c</sup> | 23.4 <sup>d</sup> |
| all_reduce | 64 KB | 6.4 | 26.8 | 4.21x | 6.3 <sup>c</sup> | 25.2 <sup>d</sup> |
| all_reduce | 1 MB | 13.8 | 28.1 | 2.05x | 13.7 <sup>a</sup> | 26.1 <sup>d</sup> |
| all_gather | 1 KB | 5.1 | 15.9 | 3.14x | 5.0 <sup>a</sup> | 15.9 <sup>c</sup> |
| all_gather | 64 KB | 6.3 | 17.6 | 2.77x | 6.2 <sup>a</sup> | 17.5 <sup>c</sup> |
| all_gather | 1 MB | 7.8 | 18.3 | 2.33x | 7.8 <sup>c</sup> | 18.2 <sup>c</sup> |
| reduce_scatter | 1 KB | 6.2 | 15.9 | 2.55x | 5.9 <sup>a</sup> | 15.6 <sup>c</sup> |
| reduce_scatter | 64 KB | 7.0 | 17.1 | 2.44x | 6.9 <sup>a</sup> | 16.9 <sup>a</sup> |
| reduce_scatter | 1 MB | 8.8 | 18.0 | 2.05x | 8.6 <sup>b</sup> | 17.8 <sup>c</sup> |
| broadcast | 8 B | 6.1 | 6.9 | 1.13x | 6.1 <sup>a</sup> | 6.3 <sup>a</sup> |
| broadcast | 1 KB | 7.8 | 7.4 | 0.95x | 7.6 <sup>c</sup> | 7.1 <sup>c</sup> |
| broadcast | 64 KB | 15.5 | 12.0 | 0.78x | 15.4 <sup>a</sup> | 11.4 <sup>c</sup> |
| broadcast | 1 MB | 42.5 | 12.3 | 0.29x | 20.1 <sup>c</sup> | 11.8 <sup>c</sup> |
| alltoall | 1 KB | 4.8 | 9.8 | 2.04x | 4.7 <sup>a</sup> | 9.1 <sup>d</sup> |
| alltoall | 64 KB | 7.6 | 11.1 | 1.45x | 7.5 <sup>b</sup> | 10.4 <sup>d</sup> |
| alltoall | 1 MB | 9.2 | 17.7 | 1.93x | 9.1 <sup>a</sup> | 16.4 <sup>d</sup> |

¹ Fastest of the fixed settings: <sup>a</sup> `NCCL_ALGO=Ring NCCL_PROTO=LL`, <sup>b</sup> `NCCL_ALGO=Tree NCCL_PROTO=LL`, <sup>c</sup> `NCCL_PROTO=LL`, <sup>d</sup> `NCCL_NVLS_ENABLE=0`.

**Run-to-run spread of the default runs** (min–max over all runs and nodes, µs):

| collective | bytes | MI355X | B200 |
|---|---:|---:|---:|
| sendrecv | 8 B | 6.9–8.1 | 8.8–9.9 |
| sendrecv | 1 KB | 8.3–8.7 | 8.8–9.8 |
| all_reduce | 8 B | 15.3–16.0 | 17.7–19.7 |
| all_reduce | 1 KB | 4.9–5.1 | 24.3–27.2 |

All nodes, settings and sizes: [`latency-1node-mi355x-rocm10.1-vs-b200.md`](latency-1node-mi355x-rocm10.1-vs-b200.md).

Sources: `../logs/node6100/rocm10.1/rccl/lat_1node_20261027_171559`, `../logs/node6101/rocm10.1/rccl/lat_1node_20261027_213705`, `../../b200-nodes/out-nccl-lat-1node/25863885`, `../../b200-nodes/out-nccl-lat-1node/25863886`. Updated 2026-10-28.
<!-- INTRA-LAT-END -->

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

**Compared: MI355X (RCCL 2.30.7, ROCm 10.1, node6100 + node6101) vs B200 (NCCL, mean of the 4 healthy node pairs), 2 nodes × 8 GPUs, one rank per GPU.** busbw (GB/s) at the largest message size (16 GiB; MI355X alltoall 4 GB). Both have 8 × 400G rails per node = 400 GB/s (§2.1). RCCL is AMD's version of NCCL and reads the same `NCCL_*` setting names, so the `NCCL_*` variables below are RCCL settings on MI355X. MI355X rows use RCCL's defaults unless a setting is named; the named setting is the fastest one tested for that collective.

| Collective | MI355X (RCCL) | MI355X ceiling | MI355X % of ceiling | B200 (NCCL) | B200 ceiling | B200 % of ceiling | MI355X / B200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_reduce | 366.4 | 400 | 92% | $\color{red}{\textsf{233.1}}$ | 400 | $\color{red}{\textsf{58\\%}}$ | $\color{red}{\textbf{1.57x}}$ |
| all_gather | 366.0 | 400 | 92% | 375.4 | 400 | 94% | 0.97x |
| reduce_scatter | 365.1 | 400 | 91% | 382.4 | 400 | 96% | 0.95x |
| broadcast | 360.0 | 400 | 90% | 359.1 | 400 | 90% | 1.00x |
| alltoall (`NCCL_NET=IB`) | 89.2 | 93.75 | 95% | $\color{red}{\textsf{48.9}}$ | 93.75 | $\color{red}{\textsf{52\\%}}$ | $\color{red}{\textbf{1.82x}}$ |
| sendrecv (`NCCL_NET=IB` + `NCCL_NCHANNELS_PER_NET_PEER=4`) | 49.4 | 50 | 99% | 48.8 | 50 | 98% | 1.01x |

Ceiling = what the network can carry for that pattern, per node and direction. **400 GB/s** for the ring/tree collectives: up to 8 rings, one per rail, together use the whole network of the node. sendrecv and alltoall use no ring, so their ceiling is what one GPU can send over its own rail:
- **alltoall, 93.75 GB/s:** each GPU sends half of its data straight to the other node over its own rail (50 GB/s), so the call runs at up to 100 GB/s; in busbw units that is 100 × 15/16. It is an upper bound: protocol overhead makes the real ceiling about 5% lower.
- **sendrecv, 50 GB/s:** each GPU pair that crosses the nodes uses one rail, and the slowest pair sets the busbw.
- **All MI355X settings tested** (busbw at the largest size): alltoall: defaults 76.7 GB/s (82%), `NCCL_NET=IB` 89.2 GB/s (95%); sendrecv: defaults 32.4 GB/s (65%), `NCCL_NET=IB` 36.5 GB/s (73%), `NCCL_NET=IB` + `NCCL_NCHANNELS_PER_NET_PEER=4` 49.4 GB/s (99%), `NCCL_NCHANNELS_PER_NET_PEER=4` 40.0 GB/s (80%).

**all_reduce, small and medium messages: RCCL's default is wrong across nodes.** With RCCL 2.30.7's defaults, all_reduce on 2 nodes × 8 GPUs returns wrong data at 256 KB (rccl-tests data check) and takes 1.4–2.2 ms from 8 B to 256 KB (about 30–50 µs is normal here). Large messages are correct and fast, so the bandwidth above is not affected. `NCCL_PROTO=LL128` gives correct results at every size and is used for all_reduce latency in §2.3. All settings, time per call in µs (✗ = wrong results at that size), data check on, 100 iterations:

| RCCL setting | sizes with wrong results | 8 | 1K | 32K | 64K | 256K | 1M | 4M |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default | 256K | 1,412.5 | 1,707.1 | 2,170.2 | 2,171.6 | 2,179.7 ✗ | 57.7 | 78.5 |
| `NCCL_NET=IB` | 256K | 1,425.8 | 1,685.5 | 2,171.5 | 2,173.0 | 2,180.3 ✗ | 58.7 | 77.3 |
| `NCCL_ALGO=Ring` | none | 53.5 | 65.6 | 68.8 | 69.1 | 79.5 | 166.6 | 166.8 |
| `NCCL_ALGO=Tree` | 128K, 256K | 1,427.6 | 1,696.3 | 2,169.9 | 2,172.1 | 2,178.9 ✗ | 57.3 | 76.0 |
| `NCCL_PROTO=Simple` | none | 77.0 | 74.8 | 88.0 | 88.6 | 91.5 | 108.0 | 156.5 |
| `NCCL_PROTO=LL` | 32K, 128K | 1,426.5 | 1,695.0 | 2,169.9 ✗ | 2,171.7 | 2,179.1 | 2,195.3 | 315.5 |
| **`NCCL_PROTO=LL128`** (used) | none | 39.8 | 39.7 | 45.8 | 45.8 | 49.7 | 57.8 | 76.8 |
| `RCCL_MSCCLPP_ENABLE=0` | 1K, 128K, 256K | 1,419.5 | 1,689.6 ✗ | 2,169.0 | 2,170.2 | 2,178.6 ✗ | 57.8 | 76.2 |
| `RCCL_MSCCL_ENABLE=0` | none | 1,405.0 | 1,692.7 | 2,169.7 | 2,171.1 | 2,178.9 | 57.8 | 77.3 |
| `NCCL_ALGO=Ring NCCL_PROTO=LL` | none | 53.7 | 64.9 | 68.2 | 69.0 | 77.9 | 124.9 | 302.2 |

> [!NOTE]
> Between two nodes, AMD Pollara (Ethernet) performs about the same as NVIDIA InfiniBand: all_gather, reduce_scatter and broadcast are 0.95–1.00x, both near the full network. MI355X is faster in all_reduce (1.57x) and alltoall (1.82x), because NCCL is low there; B200 all_reduce could improve with SHARP (data reduced inside the InfiniBand switch), which was not used here. sendrecv is 1.01x.

**Apple-to-apple: nearly.** Same test, 16 ranks, same message range, same network speed. Differences: RoCEv2 (AMD Pollara) vs InfiniBand (NVIDIA), RCCL vs NCCL. reduce, gather and scatter were not run on MI355X across 2 nodes.

**Results far from paper** (also listed in the Quick summary of [`mi355x-rocm10.1-vs-b200.md`](mi355x-rocm10.1-vs-b200.md)):

- **B200 all_reduce reaches only 58% (1.57x vs 1.00x on paper):** other NCCL collectives reach 90–96% on the same network, so **the likely cause is NCCL's all_reduce algorithm, not the network** (the algorithm NCCL picked was not logged). **SHARP should help:** the switches add the data up, so each GPU sends it once instead of twice. It could not be tested: the fabric has no SHARP Aggregation Manager (`sharp_am`).
- **B200 alltoall reaches only 52% (1.82x vs 1.00x on paper):** NCCL's alltoall sends a separate piece directly to each of the 15 other GPUs and reaches only half of the ceiling. **Likely too few channels per network peer.** Not tested on B200; `NCCL_NCHANNELS_PER_NET_PEER=4` would be the first setting to try.

<!-- LATENCY-BEGIN (auto: rccl-tests/compare_lat.py) -->
### 2.3 Measured latency (small messages)

MI355X ROCm 10.1 (RCCL 2.30.7, AMD Pollara RoCEv2) vs B200 (NCCL, NVIDIA InfiniBand). Time per call in µs, lower is better; speed = B200 time ÷ MI355X time (>1: MI355X faster). MI355X = the fastest of RCCL's default, `NCCL_NET=IB` and `NCCL_PROTO=LL128` that gives correct results (all settings in `node6100/rocm10.1/rccl_lat.md`); for all_reduce across nodes RCCL's default gives wrong results, so only `NCCL_PROTO=LL128` counts there (§2.2).

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

> **Mixed.** sendrecv: **B200 faster** at every size (0.73–0.83x); all_reduce: **B200 faster** at 8 B, 1 KB, 64 KB (0.55–0.66x), **MI355X faster** at 1 MB (1.05x).

| collective | bytes | MI355X (Pollara) | B200 (InfiniBand) | MI355X / B200 speed |
|---|---:|---:|---:|---:|
| sendrecv | 8 | 27.1 | 21.2 | 0.78x |
| sendrecv | 1K | 28.6 | 21.0 | 0.73x |
| sendrecv | 64K | 37.6 | 28.8 | 0.77x |
| sendrecv | 1M | 62.5 | 52.2 | 0.83x |
| all_reduce | 8 | 26.5 | 14.5 | 0.55x |
| all_reduce | 1K | 26.2 | 14.6 | 0.56x |
| all_reduce | 64K | 36.6 | 24.0 | 0.66x |
| all_reduce | 1M | 54.7 | 57.6 | 1.05x |

**2 nodes × 8 GPUs, network part only** (network part = 2-node time − 1-node time, same test and size; this removes the step among the 8 GPUs inside each node):

> **InfiniBand is faster** in 7 of 8 rows: Pollara is 0.25–0.65x the speed of InfiniBand in the network part. Exception: all_reduce 1 MB (1.24x).

| collective | bytes | network part, Pollara | network part, InfiniBand | Pollara / InfiniBand speed | whole call, MI355X / B200 speed |
|---|---:|---:|---:|---:|---:|
| sendrecv | 8 | 11.9 | 6.6 | 0.56x | 0.83x |
| sendrecv | 1K | 10.5 | 6.5 | 0.62x | 0.83x |
| sendrecv | 64K | 8.1 | 5.2 | 0.65x | 0.93x |
| sendrecv | 1M | 18.4 | 5.6 | 0.31x | 0.68x |
| all_reduce | 8 | 23.4 | 12.9 | 0.55x | 0.79x |
| all_reduce | 1K | 34.6 | 8.6 | 0.25x | 0.85x |
| all_reduce | 64K | 39.1 | 12.3 | 0.31x | 0.87x |
| all_reduce | 1M | 43.6 | 54.1 | 1.24x | 1.43x |

**Only in 2 nodes × 8 GPUs** (other collectives; speed only, >1: MI355X faster):

| collective | 256 B | 64 KB | 1 MB |
|---|---:|---:|---:|
| all_gather | 0.79x | 0.88x | 1.52x |
| reduce_scatter | 0.83x | 0.79x | 1.72x |
| broadcast | 0.81x | 0.37x | 1.16x |
| alltoall | 0.66x | 0.84x | 1.38x |

Sources: `logs/node6100/rocm10.1/rccl/lat_2node_20261027_215244`, `../../b200-nodes/out-nccl-lat/25690497`. Updated 2026-10-28.
<!-- LATENCY-END -->

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

**What runs on node6100/6101.** Standard RoCEv2. Checked on node6100: all 8 NICs (`ionic_0`–`ionic_7`, firmware 1.117.5-a-196) use RoCE v2 GIDs (`NCCL_IB_GID_INDEX=1`), and RCCL opens them through the normal verbs path (`NET/IB/<n>/GDRDMA`, or `NET/IB-CAST` with RCCL's default (also on ROCm 10.1)). Nothing in our runs turns on the UEC-style features, so every result in this file is plain RoCEv2.

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

