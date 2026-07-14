# nccl-tests (sendrecv) 1-node summary

- Generated: 2026-07-13 16:21:48
- Collective: sendrecv_perf
- Nodes: node5500, node5502 (each 8 x NVIDIA B200, single node, intra-node NVLink)
- Config: 1 thread, 1 MiB-16 GiB, 5 warmup + 20 iters
- Reference (MIT aicr-benchmarks, `results_b200.md`, b0027): sendrecv busbw **666 GB/s** = 74% of 900 GB/s NVLink max

## Per-node overview

| Node | Avg busbw (GB/s) | Peak busbw (GB/s) | Converged (GB/s) | % of NVLink max | % of reference | Correctness |
|------|-----------------:|------------------:|-----------------:|--------------:|-------------:|---|
| node5500 | 318.3 | 665.5 | 665.5 | 74% | 99.9% | PASS |
| node5502 | 317.2 | 664.9 | 664.9 | 74% | 99.8% | PASS |
| **reference (b0027)** | — | 666.0 | 666.0 | 74% | 100% | — |

Converged = busbw at the largest message size (16 GiB), best of out-of-place / in-place (matches the reference's methodology).

## Bus bandwidth vs message size (out-of-place busbw, GB/s)

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 28.6 | 22.6 |
| 4 MiB | 61.8 | 63.6 |
| 16 MiB | 78.0 | 77.6 |
| 64 MiB | 84.9 | 84.4 |
| 256 MiB | 332.8 | 332.3 |
| 1 GiB | 645.4 | 643.3 |
| 4 GiB | 660.8 | 660.4 |
| 16 GiB | 665.5 | 664.9 |

In-place busbw tracks out-of-place within ~2% for sendrecv. Bandwidth rises with message size and saturates at large sizes (NVLink-bound).

