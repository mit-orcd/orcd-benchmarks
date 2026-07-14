# nccl-tests 1-node collective summary

- Generated: 2026-07-14 11:05:08
- Nodes: node5500, node5502 (each 8 x NVIDIA B200, single node, intra-node NVLink)
- Config: 1 thread, 1 MiB-16 GiB, 5 warmup + 20 iters
- Collectives: sendrecv, reduce, broadcast, gather, scatter, reduce_scatter, all_gather, all_reduce, alltoall, hypercube
- Reference: MIT aicr-benchmarks `results_b200.md`, Table 1 (b0027, 8x B200, NVLink 5.0 / NVSwitch), busbw at 900 GB/s NVLink max

Converged busbw = busbw at the largest message size, best of out-of-place / in-place (matches the reference methodology). busbw (bus bandwidth) is the figure of merit.

## Converged bus bandwidth by collective (GB/s)

| Collective | node5500 | node5502 | Reference (b0027) | node5500 % of ref | node5502 % of ref | Correctness |
|---|---:|---:|---:|---:|---:|---|
| sendrecv | 665.6 | 664.8 | 666 | 100% | 100% | PASS |
| reduce | 687.6 | 685.6 | 701 | 98% | 98% | PASS |
| broadcast | 675.3 | 680.0 | 691 | 98% | 98% | PASS |
| gather | 717.7 | 716.9 | 717 | 100% | 100% | PASS |
| scatter | 734.4 | 734.7 | 746 | 98% | 98% | PASS |
| reduce_scatter | 694.9 | 694.1 | 695 | 100% | 100% | PASS |
| all_gather | 677.9 | 677.3 | 684 | 99% | 99% | PASS |
| all_reduce | 838.5 | 837.9 | 841 | 100% | 100% | PASS |
| alltoall | 661.4 | 659.5 | 675 | 98% | 98% | PASS |
| hypercube | FAILED | FAILED | — | — | — | FAIL |

## Bus bandwidth vs message size (out-of-place busbw, GB/s)

### sendrecv

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 28.0 | 23.8 |
| 4 MiB | 62.5 | 63.4 |
| 16 MiB | 78.5 | 77.6 |
| 64 MiB | 84.8 | 84.6 |
| 256 MiB | 332.6 | 332.6 |
| 1 GiB | 644.4 | 644.2 |
| 4 GiB | 661.1 | 660.7 |
| 16 GiB | 665.6 | 664.8 |

### reduce

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 25.1 | 19.5 |
| 4 MiB | 92.5 | 79.4 |
| 16 MiB | 298.4 | 278.5 |
| 64 MiB | 501.8 | 507.3 |
| 256 MiB | 619.8 | 618.7 |
| 1 GiB | 660.9 | 663.6 |
| 4 GiB | 675.4 | 677.8 |
| 16 GiB | 687.6 | 685.6 |

### broadcast

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 24.7 | 18.8 |
| 4 MiB | 84.2 | 70.6 |
| 16 MiB | 303.9 | 251.2 |
| 64 MiB | 495.1 | 493.9 |
| 256 MiB | 607.1 | 609.6 |
| 1 GiB | 642.5 | 644.4 |
| 4 GiB | 658.1 | 663.8 |
| 16 GiB | 675.3 | 679.9 |

### gather

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 23.6 | 18.6 |
| 4 MiB | 87.3 | 81.4 |
| 16 MiB | 340.7 | 320.9 |
| 64 MiB | 613.5 | 609.7 |
| 256 MiB | 683.4 | 684.3 |
| 1 GiB | 700.4 | 697.2 |
| 4 GiB | 715.1 | 715.9 |
| 16 GiB | 717.7 | 716.9 |

### scatter

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 24.3 | 19.4 |
| 4 MiB | 99.1 | 80.2 |
| 16 MiB | 378.9 | 315.5 |
| 64 MiB | 601.0 | 576.8 |
| 256 MiB | 689.4 | 637.7 |
| 1 GiB | 726.3 | 715.0 |
| 4 GiB | 730.2 | 731.0 |
| 16 GiB | 734.4 | 734.7 |

### reduce_scatter

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 18.8 | 14.9 |
| 4 MiB | 78.1 | 60.6 |
| 16 MiB | 145.5 | 143.8 |
| 64 MiB | 414.0 | 415.5 |
| 256 MiB | 587.5 | 587.8 |
| 1 GiB | 642.8 | 636.1 |
| 4 GiB | 679.8 | 680.4 |
| 16 GiB | 694.8 | 693.7 |

### all_gather

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 16.5 | 12.4 |
| 4 MiB | 70.0 | 53.1 |
| 16 MiB | 137.2 | 137.7 |
| 64 MiB | 416.0 | 414.9 |
| 256 MiB | 580.4 | 582.7 |
| 1 GiB | 616.7 | 617.9 |
| 4 GiB | 652.9 | 653.2 |
| 16 GiB | 669.0 | 669.5 |

### all_reduce

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 34.5 | 27.8 |
| 4 MiB | 126.2 | 124.3 |
| 16 MiB | 267.6 | 265.9 |
| 64 MiB | 421.0 | 419.9 |
| 256 MiB | 655.0 | 656.1 |
| 1 GiB | 732.7 | 732.5 |
| 4 GiB | 827.2 | 826.8 |
| 16 GiB | 838.5 | 836.5 |

### alltoall

| Message size | node5500 | node5502 |
|-------------:|------:|------:|
| 1 MiB | 15.2 | 11.6 |
| 4 MiB | 57.4 | 45.5 |
| 16 MiB | 222.6 | 178.7 |
| 64 MiB | 416.6 | 419.1 |
| 256 MiB | 532.3 | 531.8 |
| 1 GiB | 604.9 | 602.8 |
| 4 GiB | 647.4 | 645.8 |
| 16 GiB | 660.8 | 659.0 |

### hypercube

_No data (run failed or produced no rows)._

