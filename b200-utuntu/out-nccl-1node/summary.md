# nccl-tests 1-node collective summary

- Generated: 2026-08-10 14:40:08
- Nodes: node5700, node5701 (each 8 x NVIDIA B200, single node, intra-node NVLink)
- Config: 1 thread, 1 MiB-16 GiB, 5 warmup + 20 iters
- Collectives: sendrecv, reduce, broadcast, gather, scatter, reduce_scatter, all_gather, all_reduce, alltoall, hypercube

Converged busbw = busbw at the largest message size, best of out-of-place / in-place (matches the reference methodology). busbw (bus bandwidth) is the figure of merit.

## Converged bus bandwidth by collective (GB/s)

| Collective | node5700 | node5701 | Correctness |
|---|---:|---:|---|
| sendrecv | 655.6 | 656.2 | PASS |
| reduce | 682.2 | 701.6 | PASS |
| broadcast | 681.3 | 684.9 | PASS |
| gather | 718.1 | 718.0 | PASS |
| scatter | 746.0 | 733.5 | PASS |
| reduce_scatter | 696.1 | 695.3 | PASS |
| all_gather | 679.9 | 680.4 | PASS |
| all_reduce | 841.3 | 839.1 | PASS |
| alltoall | 661.3 | 660.0 | PASS |
| hypercube | FAILED | FAILED | FAIL |

## Bus bandwidth vs message size (out-of-place busbw, GB/s)

### sendrecv

| Message size | node5700 | node5701 |
|-------------:|------:|------:|
| 1 MiB | 30.3 | 30.3 |
| 4 MiB | 61.5 | 61.9 |
| 16 MiB | 77.1 | 76.7 |
| 64 MiB | 84.2 | 83.8 |
| 256 MiB | 329.8 | 329.4 |
| 1 GiB | 636.3 | 636.4 |
| 4 GiB | 651.2 | 652.3 |
| 16 GiB | 655.6 | 656.2 |

### reduce

| Message size | node5700 | node5701 |
|-------------:|------:|------:|
| 1 MiB | 27.0 | 27.3 |
| 4 MiB | 100.2 | 100.7 |
| 16 MiB | 316.0 | 316.5 |
| 64 MiB | 502.7 | 501.4 |
| 256 MiB | 612.0 | 610.6 |
| 1 GiB | 658.1 | 677.7 |
| 4 GiB | 674.6 | 691.1 |
| 16 GiB | 680.9 | 701.6 |

### broadcast

| Message size | node5700 | node5701 |
|-------------:|------:|------:|
| 1 MiB | 27.0 | 25.1 |
| 4 MiB | 97.0 | 91.0 |
| 16 MiB | 301.8 | 295.9 |
| 64 MiB | 496.6 | 492.2 |
| 256 MiB | 611.8 | 614.4 |
| 1 GiB | 653.2 | 653.3 |
| 4 GiB | 667.7 | 668.6 |
| 16 GiB | 679.9 | 684.5 |

### gather

| Message size | node5700 | node5701 |
|-------------:|------:|------:|
| 1 MiB | 27.3 | 27.3 |
| 4 MiB | 107.1 | 105.1 |
| 16 MiB | 413.1 | 416.1 |
| 64 MiB | 612.4 | 611.2 |
| 256 MiB | 691.3 | 692.9 |
| 1 GiB | 698.0 | 700.1 |
| 4 GiB | 717.0 | 716.9 |
| 16 GiB | 718.1 | 718.0 |

### scatter

| Message size | node5700 | node5701 |
|-------------:|------:|------:|
| 1 MiB | 26.6 | 26.7 |
| 4 MiB | 104.4 | 102.4 |
| 16 MiB | 403.8 | 346.8 |
| 64 MiB | 593.9 | 595.7 |
| 256 MiB | 665.2 | 682.9 |
| 1 GiB | 712.9 | 721.2 |
| 4 GiB | 741.2 | 729.5 |
| 16 GiB | 746.0 | 733.5 |

### reduce_scatter

| Message size | node5700 | node5701 |
|-------------:|------:|------:|
| 1 MiB | 20.9 | 20.4 |
| 4 MiB | 85.0 | 78.2 |
| 16 MiB | 143.3 | 144.1 |
| 64 MiB | 415.3 | 413.6 |
| 256 MiB | 586.1 | 587.4 |
| 1 GiB | 639.9 | 644.2 |
| 4 GiB | 680.1 | 678.6 |
| 16 GiB | 696.1 | 695.3 |

### all_gather

| Message size | node5700 | node5701 |
|-------------:|------:|------:|
| 1 MiB | 18.9 | 19.0 |
| 4 MiB | 79.0 | 79.8 |
| 16 MiB | 138.4 | 139.4 |
| 64 MiB | 414.9 | 414.5 |
| 256 MiB | 578.6 | 580.9 |
| 1 GiB | 618.0 | 621.1 |
| 4 GiB | 654.8 | 654.4 |
| 16 GiB | 669.7 | 669.2 |

### all_reduce

| Message size | node5700 | node5701 |
|-------------:|------:|------:|
| 1 MiB | 40.9 | 41.2 |
| 4 MiB | 125.7 | 126.2 |
| 16 MiB | 266.9 | 267.1 |
| 64 MiB | 421.6 | 421.5 |
| 256 MiB | 661.2 | 658.8 |
| 1 GiB | 728.7 | 728.8 |
| 4 GiB | 834.1 | 833.9 |
| 16 GiB | 841.3 | 839.1 |

### alltoall

| Message size | node5700 | node5701 |
|-------------:|------:|------:|
| 1 MiB | 16.0 | 16.3 |
| 4 MiB | 58.1 | 59.4 |
| 16 MiB | 209.5 | 234.6 |
| 64 MiB | 418.2 | 418.8 |
| 256 MiB | 526.8 | 527.3 |
| 1 GiB | 604.6 | 605.8 |
| 4 GiB | 647.0 | 646.8 |
| 16 GiB | 660.8 | 659.8 |

### hypercube

_No data (run failed or produced no rows)._

