# nccl-tests 2-node summary — Ubuntu B200 nodes

- Generated: 2026-08-11 22:14:12 — comparison restructured 2026-08-12
- Runs: node5700+node5701
- GPUs: 8/node x 2 nodes = 16 x NVIDIA B200 (inter-node, InfiniBand + GPUDirect RDMA)
- Config: 1 MiB-16 GiB, 5 warmup + 20 iters
- **busbw** (bus bandwidth) is the figure of merit throughout; results are judged against this cluster's own hardware ceiling (`HW max`), not against an external reference.
- **node5700+node5701 run Ubuntu 24.04; node5500-5502 run Rocky 8.** Sections 1-3 report the Ubuntu nodes on their own. The entire Ubuntu-vs-Rocky 8 comparison — what differs, which is better, and why — is consolidated in **section 4**. Full Rocky 8 set: `../b200-nodes/out-nccl-2node/summary.md`.

## 1. Results — bandwidth, and how close it gets to the hardware limit

A *collective* is one communication pattern that all 16 GPUs take part in together (e.g. `all_reduce` sums a buffer across every GPU; `broadcast` sends one GPU's buffer to all). The figure of merit is **busbw** (bus bandwidth, GB/s) at the largest message size. Representative node pair: **node5700+node5701**.

`HW max` is this cluster's own fabric ceiling. Each B200 owns one NDR rail at 400 Gb/s = **50 GB/s per direction** and each node has **8 rails** (mlx5_4/7/8/9/10/13/14/15, confirmed by `ibstat`), so ring and root-anchored collectives drive all 8 concurrently => **400 GB/s**, while `sendrecv` busbw is one pair's rate => **50 GB/s**. Dividing by one rail's 50 GB/s gives the effective rail count: how many of the 8 the collective actually engages.

| Collective | GPUs | busbw (GB/s) | HW max (GB/s) | % of HW max | effective rails (of 8) | verdict | correctness |
|------------|-----:|-------------:|--------------:|------------:|-----------------------:|---------|:-----------:|
| sendrecv | 16 | 48.8 | 50 | 98% | 0.98 (per pair) | at line rate | PASS |
| reduce_scatter | 16 | 380.1 | 400 | 95% | 7.60 | at fabric limit | PASS |
| reduce | 16 | 380.0 | 400 | 95% | 7.60 | at fabric limit | PASS |
| all_gather | 16 | 379.1 | 400 | 95% | 7.58 | at fabric limit | PASS |
| broadcast | 16 | 355.4 | 400 | 89% | 7.11 | at fabric limit | PASS |
| scatter | 16 | 290.5 | 400 | 73% | 5.81 | expected (root-anchored) | PASS |
| all_reduce | 16 | 268.4 | 400 | 67% | 5.37 | Ring two-pass penalty | PASS |
| gather | 16 | 92.9 | 400 | 23% | 1.86 | NCCL algorithm limit | PASS |
| alltoall | 16 | 55.4 | 400 | 14% | 1.11 | NCCL algorithm limit | PASS |

busbw at 16 GiB, best of out-of-place / in-place.

**The fabric is healthy.** `sendrecv` at 98% of one rail is the cleanest validation in the table — each GPU saturates its own rail with nothing left over. The ring collectives (`reduce_scatter`, `reduce`, `all_gather`, `broadcast`) reach 7.1-7.6 of 8 rails, NCCL running 8 parallel ring channels each crossing the node boundary on a different rail; the missing few percent is ring fill/drain and protocol overhead.

**Two shortfalls are expected.** `all_reduce` at 67% pays the Ring two-pass penalty — it fills and drains the ring twice, a fixed cost the busbw formula does not divide out. `scatter` at 73% is root-anchored and unidirectional, bound by the root GPU's own outbound capacity.

**Two are algorithm-limited, and the rail count says so precisely.** `alltoall` engages **1.1 of 8 rails** and `gather` about 1.9 — NCCL's N^2 point-to-point transfers and its fan-in to a single root are not pipelined across NICs. Neither is a fabric problem; a faster network barely helps a collective that does not use it.

> **Caveat on the denominators.** 400 GB/s is exact for the ring collectives but an *approximation* for root-anchored and all-to-all patterns, where only some traffic crosses the node boundary (for alltoall, 8 of each GPU's 15 peers are remote; the rest go over NVLink). A per-collective ceiling would shift those percentages, most likely lowering `scatter`'s. It changes no conclusion: gather and alltoall are 4-8x below any reasonable ceiling under every accounting.

## 2. Bandwidth vs message size (GB/s)

### all_gather

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 261.7 us | 3.8 | 322.3 us | 3.0 |
| 4 MiB | 326.2 us | 12.1 | 337.5 us | 11.7 |
| 16 MiB | 358.7 us | 43.9 | 376.0 us | 41.8 |
| 64 MiB | 806.7 us | 78.0 | 791.8 us | 79.5 |
| 256 MiB | 1.01 ms | 249.1 | 1.03 ms | 244.8 |
| 1 GiB | 2.88 ms | 349.7 | 2.82 ms | 356.7 |
| 4 GiB | 10.98 ms | 366.6 | 10.80 ms | 372.8 |
| 16 GiB | 43.42 ms | 370.9 | 42.49 ms | 379.1 |

### all_reduce

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 183.4 us | 10.7 | 187.9 us | 10.5 |
| 4 MiB | 902.1 us | 8.7 | 796.1 us | 9.9 |
| 16 MiB | 850.6 us | 37.0 | 841.0 us | 37.4 |
| 64 MiB | 1.70 ms | 74.2 | 1.83 ms | 68.8 |
| 256 MiB | 3.34 ms | 150.7 | 3.12 ms | 161.4 |
| 1 GiB | 10.41 ms | 193.3 | 9.07 ms | 221.9 |
| 4 GiB | 31.68 ms | 254.2 | 32.02 ms | 251.5 |
| 16 GiB | 121.94 ms | 264.2 | 120.03 ms | 268.4 |

### alltoall

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 285.3 us | 3.5 | 264.2 us | 3.7 |
| 4 MiB | 439.4 us | 8.9 | 418.4 us | 9.4 |
| 16 MiB | 827.1 us | 19.0 | 698.9 us | 22.5 |
| 64 MiB | 1.81 ms | 34.8 | 1.79 ms | 35.1 |
| 256 MiB | 6.25 ms | 40.3 | 6.65 ms | 37.9 |
| 1 GiB | 20.41 ms | 49.3 | 20.76 ms | 48.5 |
| 4 GiB | 73.83 ms | 54.5 | 74.28 ms | 54.2 |
| 16 GiB | 290.61 ms | 55.4 | 290.98 ms | 55.4 |

### broadcast

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 222.0 us | 4.7 | 241.6 us | 4.3 |
| 4 MiB | 162.4 us | 25.8 | 159.7 us | 26.3 |
| 16 MiB | 226.6 us | 74.0 | 227.0 us | 73.9 |
| 64 MiB | 416.4 us | 161.2 | 415.8 us | 161.4 |
| 256 MiB | 1.17 ms | 228.6 | 1.17 ms | 228.5 |
| 1 GiB | 3.46 ms | 310.2 | 3.50 ms | 306.7 |
| 4 GiB | 12.41 ms | 346.1 | 12.41 ms | 346.2 |
| 16 GiB | 48.41 ms | 354.9 | 48.34 ms | 355.4 |

### gather

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 46.9 us | 20.9 | 47.1 us | 20.9 |
| 4 MiB | 65.6 us | 60.0 | 66.7 us | 58.9 |
| 16 MiB | 176.6 us | 89.0 | 175.2 us | 89.8 |
| 64 MiB | 672.0 us | 93.6 | 671.8 us | 93.7 |
| 256 MiB | 2.71 ms | 93.0 | 2.70 ms | 93.1 |
| 1 GiB | 10.83 ms | 92.9 | 10.83 ms | 92.9 |
| 4 GiB | 43.35 ms | 92.9 | 43.34 ms | 92.9 |
| 16 GiB | 173.74 ms | 92.7 | 173.38 ms | 92.9 |

### reduce

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 233.1 us | 4.5 | 244.9 us | 4.3 |
| 4 MiB | 154.2 us | 27.2 | 148.6 us | 28.2 |
| 16 MiB | 213.8 us | 78.5 | 203.8 us | 82.3 |
| 64 MiB | 372.5 us | 180.1 | 372.8 us | 180.0 |
| 256 MiB | 1.09 ms | 245.3 | 1.10 ms | 243.0 |
| 1 GiB | 3.23 ms | 332.6 | 3.20 ms | 336.0 |
| 4 GiB | 11.88 ms | 361.4 | 11.63 ms | 369.2 |
| 16 GiB | 45.21 ms | 380.0 | 45.21 ms | 380.0 |

### reduce_scatter

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 265.9 us | 3.7 | 328.2 us | 3.0 |
| 4 MiB | 336.6 us | 11.7 | 380.5 us | 10.3 |
| 16 MiB | 412.2 us | 38.2 | 435.4 us | 36.1 |
| 64 MiB | 915.8 us | 68.7 | 934.6 us | 67.3 |
| 256 MiB | 1.10 ms | 228.3 | 1.11 ms | 226.3 |
| 1 GiB | 2.80 ms | 359.7 | 2.80 ms | 359.8 |
| 4 GiB | 10.74 ms | 374.9 | 10.73 ms | 375.3 |
| 16 GiB | 42.37 ms | 380.1 | 42.46 ms | 379.3 |

### scatter

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 168.5 us | 5.8 | 135.2 us | 7.3 |
| 4 MiB | 100.9 us | 39.0 | 99.2 us | 39.6 |
| 16 MiB | 135.6 us | 116.0 | 134.1 us | 117.3 |
| 64 MiB | 298.5 us | 210.8 | 281.8 us | 223.3 |
| 256 MiB | 986.3 us | 255.2 | 932.1 us | 270.0 |
| 1 GiB | 3.58 ms | 281.4 | 3.55 ms | 283.2 |
| 4 GiB | 14.00 ms | 287.6 | 13.99 ms | 287.8 |
| 16 GiB | 55.51 ms | 290.1 | 55.43 ms | 290.5 |

### sendrecv

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 50.4 us | 20.8 | 49.8 us | 21.1 |
| 4 MiB | 101.9 us | 41.1 | 101.1 us | 41.5 |
| 16 MiB | 355.9 us | 47.1 | 357.7 us | 46.9 |
| 64 MiB | 1.39 ms | 48.2 | 1.39 ms | 48.2 |
| 256 MiB | 5.55 ms | 48.4 | 5.55 ms | 48.4 |
| 1 GiB | 22.08 ms | 48.6 | 22.12 ms | 48.5 |
| 4 GiB | 88.20 ms | 48.7 | 88.27 ms | 48.7 |
| 16 GiB | 352.41 ms | 48.8 | 352.22 ms | 48.8 |

OOP = out-of-place, IP = in-place.

## 3. Network fabric

The inter-node data path on the B200 nodes is **NDR (400 Gb/s)**:

| NICs | Rate | Role |
|------|------|------|
| mlx5_4, 7, 8, 9, 10, 13, 14, 15 | **400 Gb/s (4X NDR)** | 8 GPU compute rails (active) |
| mlx5_0, 1, 2, 3 | 100 Gb/s (HDR100) | secondary (storage/mgmt) |
| mlx5_5, 6, 11, 12 | down | unused |

`nvidia_peermem` is loaded on both nodes, enabling GPUDirect RDMA so the NIC DMAs directly to/from GPU HBM over InfiniBand.

## 4. Ubuntu 24.04 vs Rocky 8

Same hardware on both sides: 8x B200 per node, 8 NDR rails per node, 16 ranks, NCCL 2.29.2. The Rocky 8 column throughout is the newest run of node5500+node5502; the other two Rocky pairs (5500+5501, 5501+5502) agree with it, so nothing below rests on a single node pair. Rocky 8 data: `../b200-nodes/out-nccl-2node/summary.md` and `../b200-nodes/notes.md`.

### 4.1 What differs, and which is better

**The Ubuntu nodes are better overall for any realistic NCCL workload, with one exception.** The differences fall into three groups, and only two of them matter.

**Large messages (16 GiB, converged busbw).** Signed difference: **+** means Ubuntu is faster.

| Collective | Ubuntu busbw (GB/s) | Rocky 8 busbw (GB/s) | difference | % of HW max (Ubuntu) |
|------------|--------------------:|---------------------:|-----------:|---------------------:|
| all_reduce | 268.4 | 233.2 | **+15.1%** | 67% |
| alltoall | 55.4 | 49.9 | **+11.1%** | 14% |
| sendrecv | 48.8 | 48.4 | +0.8% | 98% |
| gather | 92.9 | 92.9 | +0.0% | 23% |
| reduce_scatter | 380.1 | 382.4 | -0.6% | 95% |
| reduce | 380.0 | 382.7 | -0.7% | 95% |
| all_gather | 379.1 | 382.7 | -0.9% | 95% |
| broadcast | 355.4 | 361.9 | -1.8% | 89% |
| scatter | 290.5 | 327.0 | **-11.1%** | 73% |

**Everything at the fabric ceiling is a tie**, necessarily so: once a collective saturates the 8 rails the OS and driver stack cannot help. That covers the six collectives at 89-98% of `HW max` plus gather at its algorithmic plateau of 92.9 GB/s on both clusters — all within ±2%. **Ubuntu leads exactly where headroom remains**: all_reduce (+15.1%) and alltoall (+11.1%) are the only two still short of the wire at 16 GiB. all_reduce dominates data-parallel training, so this is not an academic win.

**The exception is large-message `scatter`, where Rocky 8 is better by 11%.** Ubuntu leads scatter at every size up to 1 GiB, then **plateaus at ~290 GB/s** while Rocky 8 climbs to 327. The plateau is real, not noise — it reproduces on a repeat (289.9 then 290.1 GB/s) and all three Rocky pairs land at 325-339. scatter is root-anchored, so it is bound by the root node's outbound aggregate and the Ubuntu pair hits a lower ceiling there. `gather`, root-anchored *but not* channel-split, is identical on both clusters at every size.

**Small messages (time — lower is better), and this is the difference that matters most.** Training steps issue many modest collectives, not a handful of 16 GiB ones, so per-operation cost dominates real workloads. It is also the largest and most reproducible difference in the comparison. The last three columns are Rocky/Ubuntu at 1 MiB, 16 MiB and 256 MiB; above 1.00x means Ubuntu is faster.

| Collective | Ubuntu 1 MiB (us) | Rocky 8 1 MiB (us) | 1 MiB | 16 MiB | 256 MiB |
|------------|------------------:|-------------------:|------:|-------:|--------:|
| alltoall | 285.3 | 860.9 | **3.02x** | 2.44x | 1.41x |
| all_gather | 261.7 | 580.9 | **2.22x** | 1.51x | 1.49x |
| reduce | 233.1 | 514.4 | **2.21x** | 1.45x | 1.11x |
| scatter | 168.5 | 363.8 | **2.16x** | 1.24x | 1.18x |
| all_reduce | 183.4 | 383.6 | **2.09x** | 1.31x | 1.39x |
| broadcast | 222.0 | 452.8 | **2.04x** | 0.84x | 1.19x |
| reduce_scatter | 265.9 | 504.8 | **1.90x** | 1.39x | 1.25x |
| gather | 46.9 | 51.1 | 1.09x | 1.00x | 1.00x |
| sendrecv | 50.4 | 54.4 | 1.08x | 1.00x | 0.99x |

**Two collectives are exempt at every size, and they identify the mechanism.** `sendrecv` and `gather` sit within ~1.1x from 1 MiB to 16 GiB — exactly the two NCCL does **not** split across its 8 channels (sendrecv is one contiguous chunk per pair, gather a fan-in kept on a single path). Everything that *is* channel-split shows the gap, and it decays as messages grow: averaged over those seven, **2.23x at 1 MiB, 1.45x at 16 MiB, 1.29x at 256 MiB, 1.02x at 16 GiB**. That decay is the signature of a cost paid per operation rather than per byte. (`broadcast` at 0.84x is the single dip below parity, at one size only.)

**Bulk GPUDirect path, measured directly** (`ib_write_bw`, mlx5_4, 64 MiB writes, 200 iters; Ubuntu 2026-08-10, Rocky 8 2026-07-13):

| Test | Ubuntu 5700<->5701 | Rocky 8 5500<->5502 | ratio |
|------|-------------------:|--------------------:|------:|
| host mem -> host mem | 378.5 Gb/s | 379.5 Gb/s | 1.00x |
| **NIC reads from GPU** | **395.5 Gb/s** | 147.6 Gb/s | **2.68x** |
| **NIC writes into GPU** | **379.6 Gb/s** | 286.6 Gb/s | **1.32x** |

The host-to-host row matching to within 0.3% is what makes the other two meaningful: the fabric itself is equally healthy on both clusters, and the difference is confined to the GPU-memory leg. Caveat: the Rocky 8 figure dates from 2026-07-13, and the 2026-08-06 8-GPU/node runs show no cap on this path (sendrecv 47.7-49.7 GB/s across the three pairs, matching Ubuntu's 48.8), so this one needs re-measuring before it is acted on.

**Net:** for any realistic NCCL workload the Ubuntu configuration is the better of the two, and the single regression is in a collective that rarely bottlenecks training.

### 4.2 Possible reasons — differences in system configuration

Both clusters are the same B200 platform with the same fabric, so the explanation has to lie in the software and platform configuration. What is actually different:

| Item | Ubuntu (node5700/5701) | Rocky 8 (node5500-5502) | same? |
|------|------------------------|-------------------------|-------|
| IOMMU (kernel cmdline) | `iommu=pt intel_iommu=on`, 540 groups | `iommu=pt intel_iommu=on`, 540 groups | **same** |
| NCCL | 2.29.2 | 2.29.2 | **same** |
| GPUDirect RDMA | `nvidia_peermem` loaded, DMABUF path | `nvidia_peermem` loaded, DMABUF path | **same** |
| IB rails | 8 x 400 Gb/s NDR, MTU 4096 | 8 x 400 Gb/s NDR | **same** |
| host-mem IB bandwidth (`ib_write_bw`, 64 MiB) | 378.5 Gb/s | 379.5 Gb/s | **same** |
| **MOFED / rdma-core** | OFED-internal-**25.10**-1.7.1.413 | OFED-internal-**26.04**-0.8.6 | **differs** |
| **NVIDIA driver** | **570.211.01** | **590.48.01** | **differs** |
| **CUDA (build)** | 12.9 | 13.1 | **differs** |
| **Kernel** | 6.8.0-124 on both nodes | **4.18** (5500) / **6.12** (5502) — heterogeneous | **differs** |
| PCI cmdline | `pci=realloc=off` | `pci=disable_acs_redir=pci:1000:c030` on 5502 only | differs |
| CPU / governor | Xeon Platinum 8570, `performance` | not verifiable from here | **unknown** |
| HCA firmware | 28.47.2526 | not verifiable from here | **unknown** |

**What the data already excludes.**

- *IOMMU / IOTLB pressure.* Both clusters boot the identical `iommu=pt intel_iommu=on` with the same 540 groups, and the Ubuntu pair reaches full GPUDirect line rate *under that setting* (395.5 Gb/s reading from GPU). This retires what was suspect #1 in `../b200-nodes/notes.md`.
- *NCCL version or job topology.* 2.29.2 on both, same 16-rank / 8-GPU-per-node layout.
- *One degraded node.* All three Rocky pairs show the same slow small-message times.
- *A bulk bandwidth cap explaining the small-message gap.* `sendrecv` moves its 1 MiB as one chunk over the same NIC and GPU-memory path, is identical on both clusters, and is at line rate at large sizes. A degraded bulk path would slow it too.

**The shape of the gap constrains the explanation.** The cost is paid **per network operation and per synchronisation**, not per byte — a constant overhead is a large fraction of a 1 MiB transfer and negligible in a 16 GiB one, which is exactly the observed decay. A 1 MiB all_gather becomes 8 chunks of 128 KiB across 8 channels plus cross-phase synchronisation; a 1 MiB sendrecv is one chunk. The layers that set that cost are the candidates:

1. **The InfiniBand stack — MOFED 25.10 (Ubuntu) vs 26.04 (Rocky 8).** The leading candidate: the verbs provider is precisely the layer that sets per-work-request posting cost while leaving bulk streaming untouched, which is the measured shape. Note the direction — Rocky 8 runs the *newer* stack and is slower per operation, consistent with a regression in the newer provider.
2. **Host CPU cost in NCCL's proxy thread.** That thread posts every RDMA operation on the host CPU, so its cost scales with operation *count*, not bytes. A `powersave` governor or deep C-states on the Rocky side would produce exactly this signature. Ubuntu runs `performance`; the Rocky governor has not been read. Cheapest hypothesis to test.
3. **GPU driver / CUDA — 570.211.01 + 12.9 vs 590.48.01 + 13.1.** Affects kernel launch and GDR registration, both per-operation. Weaker than (1) because it should also touch the sendrecv path, and that path is clean.
4. **Kernel — uniform 6.8 vs a heterogeneous 4.18 / 6.12 pair.** EL8's 4.18 under MOFED 26.04 is an unusual combination and its RDMA/DMABUF paths differ materially from 6.8. Not the sole cause (the 5501+5502 pair is slow too), but a live variable that should simply be removed.

**For the bulk GPUDirect difference (the 2.7x NIC-reads-from-GPU gap), the candidates are platform-level, not stack-level**, because the host-to-host row is identical while the GPU leg is not:

- **PCIe ACS state on the Broadcom switches.** With IOMMU on, Linux enables ACS on downstream ports, and ACS redirect routes peer-to-peer TLPs up to the root complex instead of straight across the switch. On node5700 it evidently costs nothing — `nvidia-smi topo -m` reports **PXB** for every GPU<->rail pair and the read is at line rate — so if it costs on Rocky 8 the difference is in BIOS/firmware ACS state or the topology itself. node5502 already carries `pci=disable_acs_redir=pci:1000:c030` and was still capped, suggesting that mask missed the switch ports actually in its GPU<->NIC path.
- **PCIe topology.** If a Rocky node reports `NODE`/`SYS` where node5700 reports `PXB`, its GPU and rail are not under a common switch and peer-to-peer goes through the host bridge — enough on its own to explain the cap.
- **PCIe Relaxed Ordering (BIOS).** Recommended by NVIDIA for GPUDirect; disabling it degrades NIC-reads-from-GPU specifically, matching the read/write asymmetry (147.6 vs 286.6) better than any symmetric explanation.
- **Max Payload Size / Max Read Request** on the HCA and GPU bridges — a smaller MPS means more TLPs per byte, again asymmetric toward reads.

**The `scatter` plateau — the one place Rocky 8 wins — has no established mechanism.** Unconfirmed possibilities: MOFED 26.04 streaming root-anchored traffic better at 16 GiB; the CUDA 13.1 build picking a different channel count or protocol than the 12.9 build; or `pci=realloc=off` leaving the Ubuntu root GPU's outbound path configured differently. `NCCL_DEBUG=INFO` on both sides would settle it — low priority, since scatter rarely bottlenecks training.

**Caveat.** The Rocky 8 rows come from `../b200-nodes/notes.md` and its run logs; those nodes are Slurm-managed and unreachable from node5700, so CPU model, governor and HCA firmware could not be compared — any of the three could matter for a per-operation cost. Deciding between the remaining candidates needs a controlled test: an `ib_write_bw` small-message sweep (many small ops vs one large op) on both clusters would separate the IB stack from everything above it.

## 5. Suggested actions

The action list derived from sections 1-4 — what to check, what to change, and in what order — is in **`../admin-nccl-notes.md`**, written as an ordered sequence of steps: the free read-only checks first (governor, whether the bulk GPUDirect path is still capped, NCCL's algorithm selection), then the PCIe/BIOS diff, then the MOFED downgrade as the first change.
