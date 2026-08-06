# nccl-tests 2-node summary (multi-collective)

- Generated: 2026-07-16 10:10:24
- Runs: node5500+node5502
- GPUs: 1/node x 2 nodes = 2 x NVIDIA B200 (inter-node, InfiniBand + GPUDirect RDMA)
- Config: 1 MiB-16 GiB, 5 warmup + 20 iters
- Reference: MIT aicr-benchmarks `results_b200.md` Table 2 (b0029+b0030, 16x B200 / NDR IB). busbw is the figure of merit.

## Per-collective busbw vs B200 reference

| Collective | GPUs | converged busbw (GB/s) | peak busbw (GB/s) | reference busbw (GB/s) | ours / ref | correctness |
|------------|-----:|-----------------------:|------------------:|-----------------------:|-----------:|:-----------:|
| sendrecv | 2 | 12.7 | 12.7 | 26.6 | 48% | PASS |

Converged = busbw at the largest message size, best of out-of-place / in-place (matches the reference methodology).

## Bus bandwidth vs message size (GB/s)

### sendrecv

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

OOP = out-of-place, IP = in-place.

## Network fabric

The inter-node data path on the B200 nodes is **NDR (400 Gb/s)**:

| NICs | Rate | Role |
|------|------|------|
| mlx5_4, 7, 8, 9, 10, 13, 14, 15 | **400 Gb/s (4X NDR)** | 8 GPU compute rails (active) |
| mlx5_0, 1, 2, 3 | 100 Gb/s (HDR100) | secondary (storage/mgmt) |
| mlx5_5, 6, 11, 12 | down | unused |

`nvidia_peermem` is loaded on both nodes, enabling GPUDirect RDMA so the NIC DMAs directly to/from GPU HBM over InfiniBand.

