# nccl-tests 1-node collective summary

- Generated: 2026-08-10 14:52:02
- Nodes: node5700, node5701 (each 8 x NVIDIA B200, single node, intra-node NVLink)
- Config: 1 thread, 1 MiB-16 GiB, 5 warmup + 20 iters
- Collectives: sendrecv, reduce, broadcast, gather, scatter, reduce_scatter, all_gather, all_reduce, alltoall, hypercube
- **node5700/node5701 run Ubuntu 24.04; node5500-5502 run Rocky 8.** One Rocky 8 node (**node5502**, newest run) is carried below as a reference column — the three Rocky nodes agree to within a few percent, so one stands in for them all. Full Rocky 8 set: `../b200-nodes/out-nccl-1node/summary.md`.

Converged busbw = busbw at the largest message size, best of out-of-place / in-place (matches the reference methodology). busbw (bus bandwidth) is the figure of merit.

## Converged bus bandwidth by collective (GB/s)

| Collective | node5700 | node5701 | node5502 (Rocky 8) | node5700 vs Rocky 8 | node5701 vs Rocky 8 | Correctness |
|---|---:|---:|---:|---:|---:|---|
| sendrecv | 655.6 | 656.2 | 664.8 | -1.4% | -1.3% | PASS |
| reduce | 682.2 | 701.6 | 671.7 | +1.6% | +4.4% | PASS |
| broadcast | 681.3 | 684.9 | 702.4 | -3.0% | -2.5% | PASS |
| gather | 718.1 | 718.0 | 700.7 | +2.5% | +2.5% | PASS |
| scatter | 746.0 | 733.5 | 749.5 | -0.5% | -2.1% | PASS |
| reduce_scatter | 696.1 | 695.3 | 715.3 | -2.7% | -2.8% | PASS |
| all_gather | 679.9 | 680.4 | 667.0 | +1.9% | +2.0% | PASS |
| all_reduce | 841.3 | 839.1 | 864.3 | -2.7% | -2.9% | PASS |
| alltoall | 661.3 | 660.0 | 678.9 | -2.6% | -2.8% | PASS |
| hypercube | FAILED | FAILED | FAILED | — | — | FAIL |

## Bus bandwidth vs message size (out-of-place busbw, GB/s)

### sendrecv

| Message size | node5700 | node5701 | node5502 (Rocky 8) | Ubuntu vs Rocky 8 |
|-------------:|------:|------:|------:|------:|
| 1 MiB | 30.3 | 30.3 | 29.0 | +4.6% |
| 4 MiB | 61.5 | 61.9 | 62.2 | -0.8% |
| 16 MiB | 77.1 | 76.7 | 78.1 | -1.5% |
| 64 MiB | 84.2 | 83.8 | 84.8 | -0.9% |
| 256 MiB | 329.8 | 329.4 | 332.1 | -0.8% |
| 1 GiB | 636.3 | 636.4 | 644.2 | -1.2% |
| 4 GiB | 651.2 | 652.3 | 660.1 | -1.3% |
| 16 GiB | 655.6 | 656.2 | 664.8 | -1.3% |

### reduce

| Message size | node5700 | node5701 | node5502 (Rocky 8) | Ubuntu vs Rocky 8 |
|-------------:|------:|------:|------:|------:|
| 1 MiB | 27.0 | 27.3 | 23.7 | +14.5% |
| 4 MiB | 100.2 | 100.7 | 88.2 | +13.9% |
| 16 MiB | 316.0 | 316.5 | 307.4 | +2.9% |
| 64 MiB | 502.7 | 501.4 | 521.1 | -3.7% |
| 256 MiB | 612.0 | 610.6 | 634.4 | -3.6% |
| 1 GiB | 658.1 | 677.7 | 682.6 | -2.2% |
| 4 GiB | 674.6 | 691.1 | 662.9 | +3.0% |
| 16 GiB | 680.9 | 701.6 | 671.4 | +3.0% |

### broadcast

| Message size | node5700 | node5701 | node5502 (Rocky 8) | Ubuntu vs Rocky 8 |
|-------------:|------:|------:|------:|------:|
| 1 MiB | 27.0 | 25.1 | 26.1 | -0.4% |
| 4 MiB | 97.0 | 91.0 | 97.3 | -3.4% |
| 16 MiB | 301.8 | 295.9 | 313.9 | -4.8% |
| 64 MiB | 496.6 | 492.2 | 514.7 | -3.9% |
| 256 MiB | 611.8 | 614.4 | 630.0 | -2.7% |
| 1 GiB | 653.2 | 653.3 | 664.3 | -1.7% |
| 4 GiB | 667.7 | 668.6 | 684.3 | -2.4% |
| 16 GiB | 679.9 | 684.5 | 701.8 | -2.8% |

### gather

| Message size | node5700 | node5701 | node5502 (Rocky 8) | Ubuntu vs Rocky 8 |
|-------------:|------:|------:|------:|------:|
| 1 MiB | 27.3 | 27.3 | 23.3 | +17.1% |
| 4 MiB | 107.1 | 105.1 | 87.2 | +21.7% |
| 16 MiB | 413.1 | 416.1 | 336.8 | +23.1% |
| 64 MiB | 612.4 | 611.2 | 597.8 | +2.3% |
| 256 MiB | 691.3 | 692.9 | 670.4 | +3.2% |
| 1 GiB | 698.0 | 700.1 | 681.1 | +2.6% |
| 4 GiB | 717.0 | 716.9 | 697.9 | +2.7% |
| 16 GiB | 718.1 | 718.0 | 700.7 | +2.5% |

### scatter

| Message size | node5700 | node5701 | node5502 (Rocky 8) | Ubuntu vs Rocky 8 |
|-------------:|------:|------:|------:|------:|
| 1 MiB | 26.6 | 26.7 | 23.3 | +14.5% |
| 4 MiB | 104.4 | 102.4 | 102.0 | +1.4% |
| 16 MiB | 403.8 | 346.8 | 392.9 | -4.5% |
| 64 MiB | 593.9 | 595.7 | 585.5 | +1.6% |
| 256 MiB | 665.2 | 682.9 | 653.0 | +3.2% |
| 1 GiB | 712.9 | 721.2 | 727.7 | -1.5% |
| 4 GiB | 741.2 | 729.5 | 745.7 | -1.4% |
| 16 GiB | 746.0 | 733.5 | 749.5 | -1.3% |

### reduce_scatter

| Message size | node5700 | node5701 | node5502 (Rocky 8) | Ubuntu vs Rocky 8 |
|-------------:|------:|------:|------:|------:|
| 1 MiB | 20.9 | 20.4 | 19.2 | +7.7% |
| 4 MiB | 85.0 | 78.2 | 79.5 | +2.6% |
| 16 MiB | 143.3 | 144.1 | 145.6 | -1.3% |
| 64 MiB | 415.3 | 413.6 | 416.6 | -0.5% |
| 256 MiB | 586.1 | 587.4 | 587.9 | -0.2% |
| 1 GiB | 639.9 | 644.2 | 636.6 | +0.8% |
| 4 GiB | 680.1 | 678.6 | 680.7 | -0.2% |
| 16 GiB | 696.1 | 695.3 | 694.9 | +0.1% |

### all_gather

| Message size | node5700 | node5701 | node5502 (Rocky 8) | Ubuntu vs Rocky 8 |
|-------------:|------:|------:|------:|------:|
| 1 MiB | 18.9 | 19.0 | 16.9 | +12.1% |
| 4 MiB | 79.0 | 79.8 | 71.4 | +11.2% |
| 16 MiB | 138.4 | 139.4 | 137.1 | +1.4% |
| 64 MiB | 414.9 | 414.5 | 410.6 | +1.0% |
| 256 MiB | 578.6 | 580.9 | 573.8 | +1.0% |
| 1 GiB | 618.0 | 621.1 | 607.2 | +2.0% |
| 4 GiB | 654.8 | 654.4 | 642.2 | +1.9% |
| 16 GiB | 669.7 | 669.2 | 659.0 | +1.6% |

### all_reduce

| Message size | node5700 | node5701 | node5502 (Rocky 8) | Ubuntu vs Rocky 8 |
|-------------:|------:|------:|------:|------:|
| 1 MiB | 40.9 | 41.2 | 37.2 | +10.2% |
| 4 MiB | 125.7 | 126.2 | 126.6 | -0.5% |
| 16 MiB | 266.9 | 267.1 | 268.8 | -0.7% |
| 64 MiB | 421.6 | 421.5 | 420.0 | +0.4% |
| 256 MiB | 661.2 | 658.8 | 658.0 | +0.3% |
| 1 GiB | 728.7 | 728.8 | 733.6 | -0.7% |
| 4 GiB | 834.1 | 833.9 | 853.1 | -2.2% |
| 16 GiB | 841.3 | 839.1 | 863.5 | -2.7% |

### alltoall

| Message size | node5700 | node5701 | node5502 (Rocky 8) | Ubuntu vs Rocky 8 |
|-------------:|------:|------:|------:|------:|
| 1 MiB | 16.0 | 16.3 | 16.1 | +0.1% |
| 4 MiB | 58.1 | 59.4 | 59.3 | -1.0% |
| 16 MiB | 209.5 | 234.6 | 231.8 | -4.2% |
| 64 MiB | 418.2 | 418.8 | 434.6 | -3.7% |
| 256 MiB | 526.8 | 527.3 | 548.2 | -3.8% |
| 1 GiB | 604.6 | 605.8 | 623.4 | -2.9% |
| 4 GiB | 647.0 | 646.8 | 665.2 | -2.8% |
| 16 GiB | 660.8 | 659.8 | 678.9 | -2.7% |

### hypercube

_No data (run failed or produced no rows)._

