# nccl-tests 2-node summary — Ubuntu B200 nodes

- Generated: 2026-08-10 17:43:23
- Runs: node5700+node5701
- GPUs: 8/node x 2 nodes = 16 x NVIDIA B200 (inter-node, InfiniBand + GPUDirect RDMA)
- Config: 1 MiB-16 GiB, 5 warmup + 20 iters
- Reference: MIT aicr-benchmarks `results_b200.md` Table 2 (b0029+b0030, 16x B200 / NDR IB). busbw is the figure of merit.
- **node5700+node5701 run Ubuntu 24.04; node5500-5502 run Rocky 8.** One Rocky 8 node pair (**node5500+node5502**, newest run) is carried in the tables below as a reference, with a signed difference column: **+** means the Ubuntu pair is faster, **-** slower. Full Rocky 8 set: `../b200-nodes/out-nccl-2node/summary.md`.

## 1. Results — bandwidth for every collective

A *collective* is one communication pattern that all 16 GPUs take part in together (e.g. `all_reduce` sums a buffer across every GPU; `broadcast` sends one GPU's buffer to all). Each row is one such pattern, measured at 1 MiB-16 GiB; the figure of merit is **busbw** (bus bandwidth, GB/s) at the largest message size.

Representative node pair: **node5700+node5701**.

| Collective | GPUs | Ubuntu busbw (GB/s) | Rocky 8 (node5500+node5502) (GB/s) | vs Rocky 8 | HW max (GB/s) | % of HW max | correctness |
|------------|-----:|-----------------------:|---------------:|-----------:|--------------:|--------------:|:-----------:|
| all_gather | 16 | 379.1 | 382.7 | -0.9% | 400 | 95% | PASS |
| all_reduce | 16 | 268.4 | 233.2 | +15.1% | 400 | 67% | PASS |
| alltoall | 16 | 55.4 | 49.9 | +11.1% | 400 | 14% | PASS |
| broadcast | 16 | 355.4 | 361.9 | -1.8% | 400 | 89% | PASS |
| gather | 16 | 92.9 | 92.9 | +0.0% | 400 | 23% | PASS |
| reduce | 16 | 380.0 | 382.7 | -0.7% | 400 | 95% | PASS |
| reduce_scatter | 16 | 380.1 | 382.4 | -0.6% | 400 | 95% | PASS |
| scatter | 16 | 290.5 | 327.0 | -11.1% | 400 | 73% | PASS |
| sendrecv | 16 | 48.8 | 48.4 | +0.8% | 50 | 98% | PASS |

Converged = busbw at the largest message size, best of out-of-place / in-place (matches the reference methodology).

`HW max` is the hardware ceiling of **this** cluster's fabric, not a figure taken from any paper. Each B200 owns one NDR rail at 400 Gb/s = **50 GB/s per direction**, and each node has **8 rails** (mlx5_4/7/8/9/10/13/14/15, confirmed by `ibstat`), so:

- **sendrecv** — busbw is one pair's rate => ceiling **50 GB/s**.
- **all other collectives** — ring/symmetric or root-anchored, driving all 8 rails concurrently => ceiling 8 x 50 = **400 GB/s** per node per direction.

The NIC is the binding constraint in both directions because PCIe Gen5 x16 is full-duplex (~63 GB/s *each* way), comfortably above the 50 GB/s rail. A collective well below its ceiling is limited by the NCCL algorithm, not by this hardware.

## 2. Why three collectives differ between Ubuntu and Rocky 8

**The dominant pattern is a latency advantage on the Ubuntu nodes that decays with message size.** At 1 MiB the Ubuntu pair leads on every collective (alltoall +203%, scatter +277%, reduce +121%, all_reduce +109%, broadcast +103%), and the gap shrinks monotonically as messages grow. That is the signature of a lower fixed per-transfer cost, not of more bandwidth.

**Collectives that reach the fabric ceiling converge.** reduce, broadcast, all_gather, reduce_scatter and sendrecv all run at 89-98% of `HW max` at 16 GiB, and there they land within +-2% of Rocky 8. Once the wire is the constraint, the OS and driver stack cannot help.

**all_reduce and alltoall are not special cases** — they are simply the only collectives that never reach the ceiling (67% and 14% of `HW max`). all_reduce is bound by its ring/tree schedule; alltoall is 15 separate peer transfers per GPU and is latency-bound throughout. With headroom left, the per-transfer advantage still shows at 16 GiB: **+15.5%** and **+11.1%**.

**scatter is the one real large-message regression.** Ubuntu leads it at small sizes (+277% at 1 MiB, +23% at 256 MiB), then **plateaus at ~290 GB/s** from 1 GiB onward while Rocky 8 climbs to 327. scatter is root-anchored — one GPU feeds all 15 peers, 8 of them remote — so it is bound by that root node's outbound aggregate, and the Ubuntu pair hits a lower ceiling there. The plateau reproduces exactly (289.9 GB/s in the sweep, 290.1 GB/s on a repeat run), so it is not run-to-run noise on this side.

Two caveats on that last point. The **mechanism is not established** — confirming it needs `NCCL_DEBUG=INFO` channel/protocol inspection on both clusters. And the Rocky 8 figure rests on a **single run** whose curve jumps oddly from 236 GB/s at 1 GiB to 318 at 4 GiB, so part of the gap may be variance in that measurement.

Note the contrast with **gather**, root-anchored like scatter: both clusters plateau at exactly 92.9 GB/s (0.0% difference). Where a structural limit binds, the two are identical — which is what makes scatter's asymmetry worth a closer look rather than dismissing it as OS noise.

**Suspected cause of the latency advantage.** The leading suspect is the platform difference documented in `../b200-nodes/notes.md`: the Rocky 8 nodes run with IOMMU enabled and have a degraded NIC-reads-from-GPU path (18.5 GB/s vs 35.8 GB/s for writes). Per-transfer address-translation overhead penalises small messages most, which matches the decaying gap. But the builds also differ (CUDA 13.1 vs 12.9) and so do the drivers (590.48.01 vs 570.211.01), so this data alone cannot attribute it to IOMMU. The clean experiment is `ib_write_bw --use_cuda` between node5700 and node5701: if these nodes show the full ~35 GB/s read path where Rocky showed 18.5, that confirms it.

## 3. How close each collective gets to the hardware limit

Dividing each result by one rail's line rate (50 GB/s) gives the most useful view: **how many of the 8 rails the collective actually engages**.

| Collective | ours / HW max | effective rails (of 8) | verdict |
|------------|--------------:|----------------:|---------|
| sendrecv | 98% | 0.98 (per pair) | at line rate |
| reduce_scatter | 95% | 7.60  | at fabric limit |
| reduce | 95% | 7.60  | at fabric limit |
| all_gather | 95% | 7.58  | at fabric limit |
| broadcast | 89% | 7.11  | at fabric limit |
| scatter | 73% | 5.81  | expected (root-anchored) |
| all_reduce | 67% | 5.37  | expected Ring two-pass penalty |
| gather | 23% | 1.86  | NCCL algorithm limit |
| alltoall | 14% | 1.11  | NCCL algorithm limit |

**At the fabric limit (89-95%).** `sendrecv` is the cleanest validation in the table: each GPU saturates its own rail, so 98% of 50 GB/s means nothing is left on the table — it is the single number that certifies the fabric is healthy. The ring collectives (`reduce_scatter`, `reduce`, `all_gather`, `broadcast`) sit at 7.1-7.6 effective rails because NCCL runs 8 parallel ring channels, each crossing the node boundary on a different rail; the missing few percent is ring fill/drain and protocol overhead, which cannot be recovered.

**Expected shortfalls.** `all_reduce` at 67% is the Ring two-pass penalty: it runs reduce_scatter then all_gather, and the busbw formula already divides out the doubled traffic (factor 2(N-1)/N), so a perfectly pipelined all_reduce would score the *same* as all_gather. It does not, because the ring fills and drains twice and pays the phase-transition latency — a fixed cost that does not shrink as bandwidth grows. `scatter` at 73% is root-anchored and unidirectional, limited by the root GPU's own outbound capacity.

**Algorithm-limited, and the numbers say so precisely.** `alltoall` at 14% engages roughly **1.1 of the 8 rails** — a literal quantification of NCCL's N^2 point-to-point transfers not being pipelined across NICs. `gather` at 23% is about 1.9 rails, the same story for fan-in to a single root. Neither is a fabric problem: a faster network barely helps a collective that does not use it.

> **Caveat on the denominators.** The 400 GB/s ceiling is exact for the ring collectives, whose traffic streams around a ring bottlenecked by its inter-node links. It is an *approximation* for the root-anchored and all-to-all patterns, where only a fraction of traffic crosses the node boundary (for alltoall, 8 of each GPU's 15 peers are remote; the rest go over NVLink). A per-collective ceiling would shift those percentages — most likely lowering `scatter`'s apparent figure. It does not change any conclusion: gather and alltoall are 4-8x below any reasonable ceiling and are algorithm-bound under every accounting.

## 4. Bandwidth vs message size (GB/s)

### all_gather

| Message size | OOP time | OOP busbw | IP time | IP busbw | Rocky 8 (node5500+node5502) OOP busbw | vs Rocky 8 |
|-------------:|---------:|----------:|--------:|---------:|---------------:|-----------:|
| 1 MiB | 261.7 us | 3.8 | 322.3 us | 3.0 | 1.7 | +122.5% |
| 4 MiB | 326.2 us | 12.1 | 337.5 us | 11.7 | 7.8 | +55.5% |
| 16 MiB | 358.7 us | 43.9 | 376.0 us | 41.8 | 29.0 | +51.4% |
| 64 MiB | 806.7 us | 78.0 | 791.8 us | 79.5 | 48.0 | +62.5% |
| 256 MiB | 1.01 ms | 249.1 | 1.03 ms | 244.8 | 166.8 | +49.4% |
| 1 GiB | 2.88 ms | 349.7 | 2.82 ms | 356.7 | 310.0 | +12.8% |
| 4 GiB | 10.98 ms | 366.6 | 10.80 ms | 372.8 | 345.8 | +6.0% |
| 16 GiB | 43.42 ms | 370.9 | 42.49 ms | 379.1 | 376.7 | -1.5% |

### all_reduce

| Message size | OOP time | OOP busbw | IP time | IP busbw | Rocky 8 (node5500+node5502) OOP busbw | vs Rocky 8 |
|-------------:|---------:|----------:|--------:|---------:|---------------:|-----------:|
| 1 MiB | 183.4 us | 10.7 | 187.9 us | 10.5 | 5.1 | +109.0% |
| 4 MiB | 902.1 us | 8.7 | 796.1 us | 9.9 | 7.5 | +16.6% |
| 16 MiB | 850.6 us | 37.0 | 841.0 us | 37.4 | 28.2 | +31.1% |
| 64 MiB | 1.70 ms | 74.2 | 1.83 ms | 68.8 | 53.0 | +39.8% |
| 256 MiB | 3.34 ms | 150.7 | 3.12 ms | 161.4 | 108.2 | +39.4% |
| 1 GiB | 10.41 ms | 193.3 | 9.07 ms | 221.9 | 156.4 | +23.6% |
| 4 GiB | 31.68 ms | 254.2 | 32.02 ms | 251.5 | 203.1 | +25.1% |
| 16 GiB | 121.94 ms | 264.2 | 120.03 ms | 268.4 | 228.8 | +15.5% |

### alltoall

| Message size | OOP time | OOP busbw | IP time | IP busbw | Rocky 8 (node5500+node5502) OOP busbw | vs Rocky 8 |
|-------------:|---------:|----------:|--------:|---------:|---------------:|-----------:|
| 1 MiB | 285.3 us | 3.5 | 264.2 us | 3.7 | 1.1 | +202.6% |
| 4 MiB | 439.4 us | 8.9 | 418.4 us | 9.4 | 4.0 | +121.5% |
| 16 MiB | 827.1 us | 19.0 | 698.9 us | 22.5 | 7.8 | +144.2% |
| 64 MiB | 1.81 ms | 34.8 | 1.79 ms | 35.1 | 21.4 | +62.5% |
| 256 MiB | 6.25 ms | 40.3 | 6.65 ms | 37.9 | 28.5 | +41.4% |
| 1 GiB | 20.41 ms | 49.3 | 20.76 ms | 48.5 | 38.9 | +26.7% |
| 4 GiB | 73.83 ms | 54.5 | 74.28 ms | 54.2 | 46.9 | +16.3% |
| 16 GiB | 290.61 ms | 55.4 | 290.98 ms | 55.4 | 49.9 | +11.1% |

### broadcast

| Message size | OOP time | OOP busbw | IP time | IP busbw | Rocky 8 (node5500+node5502) OOP busbw | vs Rocky 8 |
|-------------:|---------:|----------:|--------:|---------:|---------------:|-----------:|
| 1 MiB | 222.0 us | 4.7 | 241.6 us | 4.3 | 2.3 | +103.4% |
| 4 MiB | 162.4 us | 25.8 | 159.7 us | 26.3 | 20.6 | +25.6% |
| 16 MiB | 226.6 us | 74.0 | 227.0 us | 73.9 | 87.7 | -15.6% |
| 64 MiB | 416.4 us | 161.2 | 415.8 us | 161.4 | 160.1 | +0.6% |
| 256 MiB | 1.17 ms | 228.6 | 1.17 ms | 228.5 | 192.2 | +18.9% |
| 1 GiB | 3.46 ms | 310.2 | 3.50 ms | 306.7 | 224.7 | +38.0% |
| 4 GiB | 12.41 ms | 346.1 | 12.41 ms | 346.2 | 312.6 | +10.7% |
| 16 GiB | 48.41 ms | 354.9 | 48.34 ms | 355.4 | 360.0 | -1.4% |

### gather

| Message size | OOP time | OOP busbw | IP time | IP busbw | Rocky 8 (node5500+node5502) OOP busbw | vs Rocky 8 |
|-------------:|---------:|----------:|--------:|---------:|---------------:|-----------:|
| 1 MiB | 46.9 us | 20.9 | 47.1 us | 20.9 | 19.2 | +8.8% |
| 4 MiB | 65.6 us | 60.0 | 66.7 us | 58.9 | 57.0 | +5.1% |
| 16 MiB | 176.6 us | 89.0 | 175.2 us | 89.8 | 89.2 | -0.2% |
| 64 MiB | 672.0 us | 93.6 | 671.8 us | 93.7 | 93.2 | +0.5% |
| 256 MiB | 2.71 ms | 93.0 | 2.70 ms | 93.1 | 93.0 | +0.1% |
| 1 GiB | 10.83 ms | 92.9 | 10.83 ms | 92.9 | 92.9 | +0.0% |
| 4 GiB | 43.35 ms | 92.9 | 43.34 ms | 92.9 | 92.9 | +0.0% |
| 16 GiB | 173.74 ms | 92.7 | 173.38 ms | 92.9 | 92.9 | -0.2% |

### reduce

| Message size | OOP time | OOP busbw | IP time | IP busbw | Rocky 8 (node5500+node5502) OOP busbw | vs Rocky 8 |
|-------------:|---------:|----------:|--------:|---------:|---------------:|-----------:|
| 1 MiB | 233.1 us | 4.5 | 244.9 us | 4.3 | 2.0 | +120.6% |
| 4 MiB | 154.2 us | 27.2 | 148.6 us | 28.2 | 16.2 | +68.0% |
| 16 MiB | 213.8 us | 78.5 | 203.8 us | 82.3 | 54.2 | +44.7% |
| 64 MiB | 372.5 us | 180.1 | 372.8 us | 180.0 | 139.4 | +29.2% |
| 256 MiB | 1.09 ms | 245.3 | 1.10 ms | 243.0 | 220.3 | +11.4% |
| 1 GiB | 3.23 ms | 332.6 | 3.20 ms | 336.0 | 270.5 | +23.0% |
| 4 GiB | 11.88 ms | 361.4 | 11.63 ms | 369.2 | 332.5 | +8.7% |
| 16 GiB | 45.21 ms | 380.0 | 45.21 ms | 380.0 | 372.6 | +2.0% |

### reduce_scatter

| Message size | OOP time | OOP busbw | IP time | IP busbw | Rocky 8 (node5500+node5502) OOP busbw | vs Rocky 8 |
|-------------:|---------:|----------:|--------:|---------:|---------------:|-----------:|
| 1 MiB | 265.9 us | 3.7 | 328.2 us | 3.0 | 1.9 | +89.7% |
| 4 MiB | 336.6 us | 11.7 | 380.5 us | 10.3 | 7.4 | +58.3% |
| 16 MiB | 412.2 us | 38.2 | 435.4 us | 36.1 | 27.5 | +38.9% |
| 64 MiB | 915.8 us | 68.7 | 934.6 us | 67.3 | 48.3 | +42.3% |
| 256 MiB | 1.10 ms | 228.3 | 1.11 ms | 226.3 | 182.5 | +25.1% |
| 1 GiB | 2.80 ms | 359.7 | 2.80 ms | 359.8 | 302.1 | +19.1% |
| 4 GiB | 10.74 ms | 374.9 | 10.73 ms | 375.3 | 357.8 | +4.8% |
| 16 GiB | 42.37 ms | 380.1 | 42.46 ms | 379.3 | 376.8 | +0.9% |

### scatter

| Message size | OOP time | OOP busbw | IP time | IP busbw | Rocky 8 (node5500+node5502) OOP busbw | vs Rocky 8 |
|-------------:|---------:|----------:|--------:|---------:|---------------:|-----------:|
| 1 MiB | 168.5 us | 5.8 | 135.2 us | 7.3 | 2.7 | +115.9% |
| 4 MiB | 100.9 us | 39.0 | 99.2 us | 39.6 | 11.1 | +251.3% |
| 16 MiB | 135.6 us | 116.0 | 134.1 us | 117.3 | 93.9 | +23.5% |
| 64 MiB | 298.5 us | 210.8 | 281.8 us | 223.3 | 193.4 | +9.0% |
| 256 MiB | 986.3 us | 255.2 | 932.1 us | 270.0 | 215.8 | +18.3% |
| 1 GiB | 3.58 ms | 281.4 | 3.55 ms | 283.2 | 236.2 | +19.1% |
| 4 GiB | 14.00 ms | 287.6 | 13.99 ms | 287.8 | 318.0 | -9.6% |
| 16 GiB | 55.51 ms | 290.1 | 55.43 ms | 290.5 | 327.0 | -11.3% |

### sendrecv

| Message size | OOP time | OOP busbw | IP time | IP busbw | Rocky 8 (node5500+node5502) OOP busbw | vs Rocky 8 |
|-------------:|---------:|----------:|--------:|---------:|---------------:|-----------:|
| 1 MiB | 50.4 us | 20.8 | 49.8 us | 21.1 | 19.3 | +7.9% |
| 4 MiB | 101.9 us | 41.1 | 101.1 us | 41.5 | 41.0 | +0.4% |
| 16 MiB | 355.9 us | 47.1 | 357.7 us | 46.9 | 47.2 | -0.2% |
| 64 MiB | 1.39 ms | 48.2 | 1.39 ms | 48.2 | 48.4 | -0.3% |
| 256 MiB | 5.55 ms | 48.4 | 5.55 ms | 48.4 | 48.7 | -0.6% |
| 1 GiB | 22.08 ms | 48.6 | 22.12 ms | 48.5 | 48.9 | -0.5% |
| 4 GiB | 88.20 ms | 48.7 | 88.27 ms | 48.7 | 46.4 | +5.0% |
| 16 GiB | 352.41 ms | 48.8 | 352.22 ms | 48.8 | 47.7 | +2.2% |

OOP = out-of-place, IP = in-place.

## 5. Network fabric

The inter-node data path on the B200 nodes is **NDR (400 Gb/s)**:

| NICs | Rate | Role |
|------|------|------|
| mlx5_4, 7, 8, 9, 10, 13, 14, 15 | **400 Gb/s (4X NDR)** | 8 GPU compute rails (active) |
| mlx5_0, 1, 2, 3 | 100 Gb/s (HDR100) | secondary (storage/mgmt) |
| mlx5_5, 6, 11, 12 | down | unused |

`nvidia_peermem` is loaded on both nodes, enabling GPUDirect RDMA so the NIC DMAs directly to/from GPU HBM over InfiniBand.

