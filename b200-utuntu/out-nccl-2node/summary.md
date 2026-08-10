# nccl-tests 2-node summary (multi-collective)

- Generated: 2026-08-10 17:16:35
- Runs: node5700+node5701
- GPUs: 8/node x 2 nodes = 16 x NVIDIA B200 (inter-node, InfiniBand + GPUDirect RDMA)
- Config: 1 MiB-16 GiB, 5 warmup + 20 iters
- Reference: MIT aicr-benchmarks `results_b200.md` Table 2 (b0029+b0030, 16x B200 / NDR IB). busbw is the figure of merit.
- **node5700+node5701 run Ubuntu 24.04; node5500-5502 run Rocky 8.** One Rocky 8 node pair (**node5500+node5502**, newest run) is carried in the tables below as a reference, with a signed difference column: **+** means the Ubuntu pair is faster, **-** slower. Full Rocky 8 set: `../b200-nodes/out-nccl-2node/summary.md`.

## Per-collective busbw vs B200 reference

Representative node pair: **node5700+node5701**.

| Collective | GPUs | converged busbw (GB/s) | peak busbw (GB/s) | Rocky 8 (node5500+node5502) busbw (GB/s) | vs Rocky 8 | reference busbw (GB/s) | ours / ref | HW max (GB/s) | ours / HW max | correctness |
|------------|-----:|-----------------------:|------------------:|---------------:|-----------:|-----------------------:|-----------:|--------------:|--------------:|:-----------:|
| scatter | 16 | 290.5 | 290.5 | 327.0 | -11.1% | — | — | 400 | 73% | PASS |

Converged = busbw at the largest message size, best of out-of-place / in-place (matches the reference methodology).

`HW max` is the hardware ceiling of **this** cluster's fabric, not a figure taken from any paper. Each B200 owns one NDR rail at 400 Gb/s = **50 GB/s per direction**, and each node has **8 rails** (mlx5_4/7/8/9/10/13/14/15, confirmed by `ibstat`), so:

- **sendrecv** — busbw is one pair's rate => ceiling **50 GB/s**.
- **all other collectives** — ring/symmetric or root-anchored, driving all 8 rails concurrently => ceiling 8 x 50 = **400 GB/s** per node per direction.

The NIC is the binding constraint in both directions because PCIe Gen5 x16 is full-duplex (~63 GB/s *each* way), comfortably above the 50 GB/s rail. A collective well below its ceiling is limited by the NCCL algorithm, not by this hardware.

## Interpreting `ours / HW max`

Dividing each result by one rail's line rate (50 GB/s) gives the most useful view: **how many of the 8 rails the collective actually engages**.

| Collective | ours / HW max | effective rails (of 8) | verdict |
|------------|--------------:|----------------:|---------|
| scatter | 73% | 5.81  | expected (root-anchored) |

**At the hardware limit (92-99%).** `sendrecv` is the cleanest validation in the table: each GPU saturates its own rail, so ~99% of 50 GB/s means nothing is left on the table. It is the single number that certifies the fabric is healthy. The ring collectives sit at 7.3-7.5 effective rails because NCCL runs 8 parallel ring channels, each crossing the node boundary on a different rail; the missing few percent is ring fill/drain and protocol overhead, which cannot be recovered.

**Expected shortfalls.** `all_reduce` at ~60% is the Ring two-pass penalty: it runs reduce_scatter then all_gather, and the busbw formula already divides out the doubled traffic (factor 2(N-1)/N), so a perfectly pipelined all_reduce would score the *same* as all_gather. It does not, because the ring fills and drains twice and pays the phase-transition latency. That fixed latency does not shrink when bandwidth grows, which is why our all_reduce is a *smaller* fraction of our all_gather (~65%) than the reference's was (~78%) — and why SHARP, which collapses the two passes into one in-switch reduction, has more to gain here (see `out-nccl-2node-sharp/`). `scatter` is root-anchored and unidirectional, limited by the root's own outbound capacity.

**Algorithm-limited, and the numbers say so precisely.** `alltoall` at ~12% is exactly 1/8 — it engages roughly **one rail's worth** of bandwidth out of 8, a literal quantification of NCCL's N^2 point-to-point transfers not being pipelined across NICs. `gather` at ~24% is about two rails, the same story for fan-in to a single root. The decisive evidence that these are algorithmic rather than physical: this fabric is ~1.9x faster than the reference on sendrecv, yet gather improved only 1.05x and alltoall 1.19x. A faster fabric barely helps a collective that is not using it.

> **Caveat on the denominators.** The 400 GB/s ceiling is exact for the ring collectives, whose traffic streams around a ring bottlenecked by its inter-node links. It is an *approximation* for the root-anchored and all-to-all patterns, where only a fraction of traffic crosses the node boundary (for alltoall, 8 of each GPU's 15 peers are remote; the rest go over NVLink). A per-collective ceiling would shift those percentages — most likely lowering `scatter`'s apparent figure. It does not change any conclusion: gather and alltoall are 4-8x below any reasonable ceiling and are algorithm-bound under every accounting.

## Bus bandwidth vs message size (GB/s)

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

OOP = out-of-place, IP = in-place.

## Why the Ubuntu and Rocky 8 results differ

**The dominant pattern is a latency advantage on the Ubuntu nodes that decays with message size.** At 1 MiB the Ubuntu pair leads on every collective (alltoall +203%, scatter +277%, reduce +121%, all_reduce +109%, broadcast +103%), and the gap shrinks monotonically as messages grow. That is the signature of a lower fixed per-transfer cost, not of more bandwidth.

**Collectives that reach the fabric ceiling converge.** reduce, broadcast, all_gather, reduce_scatter and sendrecv all run at 89-98% of `HW max` at 16 GiB, and there they land within +-2% of Rocky 8. Once the wire is the constraint, the OS and driver stack cannot help.

**all_reduce and alltoall are not special cases** — they are simply the only collectives that never reach the ceiling (67% and 14% of `HW max`). all_reduce is bound by its ring/tree schedule; alltoall is 15 separate peer transfers per GPU and is latency-bound throughout. With headroom left, the per-transfer advantage still shows at 16 GiB: **+15.5%** and **+11.1%**.

**scatter is the one real large-message regression.** Ubuntu leads it at small sizes (+277% at 1 MiB, +23% at 256 MiB), then **plateaus at ~290 GB/s** from 1 GiB onward while Rocky 8 climbs to 327. scatter is root-anchored — one GPU feeds all 15 peers, 8 of them remote — so it is bound by that root node's outbound aggregate, and the Ubuntu pair hits a lower ceiling there. The plateau reproduces exactly (289.9 GB/s in the sweep, 290.1 GB/s on a repeat run), so it is not run-to-run noise on this side.

Two caveats on that last point. The **mechanism is not established** — confirming it needs `NCCL_DEBUG=INFO` channel/protocol inspection on both clusters. And the Rocky 8 figure rests on a **single run** whose curve jumps oddly from 236 GB/s at 1 GiB to 318 at 4 GiB, so part of the gap may be variance in that measurement.

Note the contrast with **gather**, root-anchored like scatter: both clusters plateau at exactly 92.9 GB/s (0.0% difference). Where a structural limit binds, the two are identical — which is what makes scatter's asymmetry worth a closer look rather than dismissing it as OS noise.

**Suspected cause of the latency advantage.** The leading suspect is the platform difference documented in `../b200-nodes/notes.md`: the Rocky 8 nodes run with IOMMU enabled and have a degraded NIC-reads-from-GPU path (18.5 GB/s vs 35.8 GB/s for writes). Per-transfer address-translation overhead penalises small messages most, which matches the decaying gap. But the builds also differ (CUDA 13.1 vs 12.9) and so do the drivers (590.48.01 vs 570.211.01), so this data alone cannot attribute it to IOMMU. The clean experiment is `ib_write_bw --use_cuda` between node5700 and node5701: if these nodes show the full ~35 GB/s read path where Rocky showed 18.5, that confirms it.

## Network fabric

The inter-node data path on the B200 nodes is **NDR (400 Gb/s)**:

| NICs | Rate | Role |
|------|------|------|
| mlx5_4, 7, 8, 9, 10, 13, 14, 15 | **400 Gb/s (4X NDR)** | 8 GPU compute rails (active) |
| mlx5_0, 1, 2, 3 | 100 Gb/s (HDR100) | secondary (storage/mgmt) |
| mlx5_5, 6, 11, 12 | down | unused |

`nvidia_peermem` is loaded on both nodes, enabling GPUDirect RDMA so the NIC DMAs directly to/from GPU HBM over InfiniBand.

