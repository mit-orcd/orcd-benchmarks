# nccl-tests (sendrecv) 2-node summary

- Generated: 2026-07-13 16:52:38
- Collective: sendrecv_perf  (inter-node, InfiniBand + GPUDirect RDMA)
- Runs: node5500+node5502 (2 nodes x 1 NVIDIA B200, one GPU per node)
- Config: 1 thread, 1 MiB-16 GiB, 5 warmup + 20 iters
- Reference (MIT aicr-benchmarks, `results_b200.md` Table 2, b0029+b0030): sendrecv busbw **26.6 GB/s** = 100% of the 26.7 GB/s GDRDMA bidir per-pair ceiling

## Overview

| Run | Avg busbw (GB/s) | Peak busbw (GB/s) | Converged (GB/s) | % of GDRDMA ceiling | % of reference | Correctness |
|-----|-----------------:|------------------:|-----------------:|----------------:|-------------:|---|
| node5500+node5502 | 12.0 | 12.7 | 12.7 | 48% | 48% | PASS |
| **reference (b0029+b0030)** | — | 26.6 | 26.6 | 100% | 100% | — |

Converged = busbw at the largest message size (16 GiB), best of out-of-place / in-place (matches the reference's methodology). The GDRDMA bidir per-pair ceiling (~26.7 GB/s) is the B200 hardware limit for a single cross-node GPU pair: one PCIe Gen5 x16 DMA engine shared between simultaneous TX+RX.

## Bus bandwidth vs message size (GB/s)

| Message size | OOP time | OOP busbw | IP time | IP busbw |
|-------------:|---------:|----------:|--------:|---------:|
| 1 MiB | 117.7 us | 8.9 | 115.3 us | 9.1 |
| 4 MiB | 366.4 us | 11.4 | 366.1 us | 11.5 |
| 16 MiB | 1.35 ms | 12.4 | 1.35 ms | 12.4 |
| 64 MiB | 5.29 ms | 12.7 | 5.29 ms | 12.7 |
| 256 MiB | 21.12 ms | 12.7 | 21.14 ms | 12.7 |
| 1 GiB | 84.45 ms | 12.7 | 84.55 ms | 12.7 |
| 4 GiB | 337.86 ms | 12.7 | 338.01 ms | 12.7 |
| 16 GiB | 1351.57 ms | 12.7 | 1351.68 ms | 12.7 |

OOP = out-of-place, IP = in-place. Bandwidth rises with message size and saturates once the per-pair GDRDMA DMA budget is the binding constraint.

## Network fabric

`ibstat` on **both node5500 and node5502** — the inter-node data path is **NDR (400 Gb/s), not HDR**:

| NICs | Rate | Role |
|------|------|------|
| mlx5_4, 7, 8, 9, 10, 13, 14, 15 | **400 Gb/s (4X NDR)** | 8 GPU compute rails (active) |
| mlx5_0, 1, 2, 3 | 100 Gb/s (2X HDR / HDR100) | secondary (storage/mgmt), active |
| mlx5_5, 6, 11, 12 | down (SDR/QDR placeholder) | unused |

The sendrecv run bound to `mlx5_4` (NDR 400 Gb/s) on both nodes, so the ~12.7 GB/s ceiling is **not** a network-rate limit — a single 400 Gb/s NDR link carries ~50 GB/s per direction, well above what was achieved. The HDR100 NICs (mlx5_0-3) are not on the NCCL data path.

## Comparison to B200 reference (sendrecv, 2-node)

| Metric | This run | Reference (b0029+b0030) |
|--------|---------:|------------------------:|
| Converged busbw (GB/s) | 12.7 | 26.6 |
| % of GDRDMA ceiling (26.7 GB/s) | 48% | 100% |
| This run / reference | 48% | 100% |

> The measured 12.7 GB/s is ~48% of the reference and ~48% of the 26.7 GB/s per-pair hardware ceiling — a clean factor of ~2 short. Diagnostics on node5500/node5502 rule out the usual suspects: the GPU-facing NICs are **NDR** (mlx5_4 Active, Rate 400 Gb/s — not HDR), `nvidia_peermem` is loaded on both nodes, the GPU runs **PCIe Gen5 x16**, and NCCL uses a GDRDMA path with good GPU-NIC affinity (PXB, same PCIe switch). Installing `nvidia_peermem` on node5500 did **not** change the result (still 12.7 GB/s across reruns), so GDR/peermem is not the bottleneck. The leading remaining hypothesis is that a single cross-node GPU pair over one NIC/QP does not saturate the ~26.7 GB/s bidirectional PCIe-DMA budget the reference figure assumes. Next steps: (1) re-test at **8 GPUs/node** so all 8 NICs are active and measure the aggregate fabric; (2) try `NCCL_IB_QPS_PER_CONNECTION=4` and more channels to add concurrency on the single pair.

