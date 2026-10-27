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
| **Inter-node bandwidth** (2 nodes, large messages), §2.2 | pending | pending |
| **Inter-node latency** (2 nodes, small messages), §2.3 | pending | pending |

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

Sources: `../logs/node6100/rocm10.1/rccl/lat_1node_20261027_171559`, `../logs/node6101/rocm10.1/rccl/lat_1node_20261027_213705`, `../../b200-nodes/out-nccl-lat-1node/25863885`, `../../b200-nodes/out-nccl-lat-1node/25863886`. Updated 2026-10-27.
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

_Pending: the 2-node runs on ROCm 10.1 (bandwidth with all settings, all_reduce data check) are running._

<!-- LATENCY-BEGIN (auto: rccl-tests/compare_lat.py) -->
### 2.3 Measured latency (small messages)

_Pending: the 2-node latency runs on ROCm 10.1 are running._
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

