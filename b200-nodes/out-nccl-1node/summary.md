# nccl-tests 1-node collective summary

- Generated: 2026-08-06 17:03:17
- Nodes: node5500, node5501, node5502 (each 8 x NVIDIA B200, single node, intra-node NVLink)
- Config: 1 thread, 1 MiB-16 GiB, 5 warmup + 20 iters
- Collectives: sendrecv, reduce, broadcast, gather, scatter, reduce_scatter, all_gather, all_reduce, alltoall, hypercube
- Reference: MIT aicr-benchmarks `results_b200.md`, Table 1 (b0027, 8x B200, NVLink 5.0 / NVSwitch), busbw at 900 GB/s NVLink max

Converged busbw = busbw at the largest message size, best of out-of-place / in-place (matches the reference methodology). busbw (bus bandwidth) is the figure of merit.

## Converged bus bandwidth by collective (GB/s)

| Collective | node5500 | node5501 | node5502 | Reference (b0027) | node5500 % of ref | node5501 % of ref | node5502 % of ref | Correctness |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| sendrecv | 695.0 | 664.5 | 664.8 | 666 | 104% | 100% | 100% | PASS |
| reduce | 673.7 | 685.8 | 671.7 | 701 | 96% | 98% | 96% | PASS |
| broadcast | 707.7 | 677.6 | 702.4 | 691 | 102% | 98% | 102% | PASS |
| gather | 755.7 | 717.8 | 700.7 | 717 | 105% | 100% | 98% | PASS |
| scatter | 705.7 | 734.3 | 749.5 | 746 | 95% | 98% | 100% | PASS |
| reduce_scatter | 667.2 | 691.3 | 715.3 | 695 | 96% | 99% | 103% | PASS |
| all_gather | 721.3 | 673.2 | 667.0 | 684 | 105% | 98% | 98% | PASS |
| all_reduce | 797.2 | 837.5 | 864.3 | 841 | 95% | 100% | 103% | PASS |
| alltoall | 648.5 | 659.9 | 678.9 | 675 | 96% | 98% | 101% | PASS |
| hypercube | FAILED | FAILED | FAILED | — | — | — | — | FAIL |

## Bus bandwidth vs message size (out-of-place busbw, GB/s)

### sendrecv

| Message size | node5500 | node5501 | node5502 |
|-------------:|------:|------:|------:|
| 1 MiB | 26.7 | 26.4 | 29.0 |
| 4 MiB | 61.6 | 62.6 | 62.2 |
| 16 MiB | 76.2 | 77.7 | 78.1 |
| 64 MiB | 88.5 | 84.5 | 84.8 |
| 256 MiB | 347.2 | 332.4 | 332.1 |
| 1 GiB | 672.8 | 644.0 | 644.2 |
| 4 GiB | 690.8 | 660.4 | 660.1 |
| 16 GiB | 695.0 | 664.5 | 664.8 |

### reduce

| Message size | node5500 | node5501 | node5502 |
|-------------:|------:|------:|------:|
| 1 MiB | 26.6 | 23.1 | 23.7 |
| 4 MiB | 99.6 | 85.3 | 88.2 |
| 16 MiB | 326.1 | 298.9 | 307.4 |
| 64 MiB | 524.4 | 506.8 | 521.1 |
| 256 MiB | 645.8 | 619.6 | 634.4 |
| 1 GiB | 689.5 | 663.7 | 682.6 |
| 4 GiB | 704.5 | 674.5 | 662.9 |
| 16 GiB | 672.5 | 683.9 | 671.4 |

### broadcast

| Message size | node5500 | node5501 | node5502 |
|-------------:|------:|------:|------:|
| 1 MiB | 24.7 | 26.0 | 26.1 |
| 4 MiB | 84.3 | 94.5 | 97.3 |
| 16 MiB | 294.7 | 305.4 | 313.9 |
| 64 MiB | 517.0 | 496.5 | 514.7 |
| 256 MiB | 638.2 | 611.1 | 630.0 |
| 1 GiB | 675.5 | 643.9 | 664.3 |
| 4 GiB | 690.4 | 659.9 | 684.3 |
| 16 GiB | 707.7 | 677.6 | 701.8 |

### gather

| Message size | node5500 | node5501 | node5502 |
|-------------:|------:|------:|------:|
| 1 MiB | 23.3 | 24.7 | 23.3 |
| 4 MiB | 91.2 | 95.5 | 87.2 |
| 16 MiB | 340.9 | 395.7 | 336.8 |
| 64 MiB | 646.5 | 612.6 | 597.8 |
| 256 MiB | 733.1 | 690.2 | 670.4 |
| 1 GiB | 737.0 | 702.3 | 681.1 |
| 4 GiB | 753.0 | 715.3 | 697.9 |
| 16 GiB | 755.6 | 717.7 | 700.7 |

### scatter

| Message size | node5500 | node5501 | node5502 |
|-------------:|------:|------:|------:|
| 1 MiB | 26.0 | 25.1 | 23.3 |
| 4 MiB | 92.4 | 99.7 | 102.0 |
| 16 MiB | 360.6 | 382.1 | 392.9 |
| 64 MiB | 628.1 | 595.3 | 585.5 |
| 256 MiB | 724.5 | 690.1 | 653.0 |
| 1 GiB | 765.5 | 727.0 | 727.7 |
| 4 GiB | 701.7 | 730.2 | 745.7 |
| 16 GiB | 705.7 | 734.3 | 749.5 |

### reduce_scatter

| Message size | node5500 | node5501 | node5502 |
|-------------:|------:|------:|------:|
| 1 MiB | 20.6 | 18.5 | 19.2 |
| 4 MiB | 84.3 | 66.1 | 79.5 |
| 16 MiB | 155.2 | 144.9 | 145.6 |
| 64 MiB | 443.1 | 416.4 | 416.6 |
| 256 MiB | 625.3 | 585.0 | 587.9 |
| 1 GiB | 683.5 | 642.3 | 636.6 |
| 4 GiB | 723.4 | 676.8 | 680.7 |
| 16 GiB | 666.6 | 689.6 | 694.9 |

### all_gather

| Message size | node5500 | node5501 | node5502 |
|-------------:|------:|------:|------:|
| 1 MiB | 16.8 | 15.7 | 16.9 |
| 4 MiB | 71.2 | 65.1 | 71.4 |
| 16 MiB | 136.2 | 137.1 | 137.1 |
| 64 MiB | 407.5 | 416.0 | 410.6 |
| 256 MiB | 568.3 | 579.8 | 573.8 |
| 1 GiB | 603.5 | 616.3 | 607.2 |
| 4 GiB | 639.7 | 649.1 | 642.2 |
| 16 GiB | 656.2 | 665.6 | 659.0 |

### all_reduce

| Message size | node5500 | node5501 | node5502 |
|-------------:|------:|------:|------:|
| 1 MiB | 35.4 | 34.1 | 37.2 |
| 4 MiB | 119.6 | 125.3 | 126.6 |
| 16 MiB | 252.8 | 266.1 | 268.8 |
| 64 MiB | 399.0 | 422.2 | 420.0 |
| 256 MiB | 622.0 | 659.1 | 658.0 |
| 1 GiB | 697.1 | 730.6 | 733.6 |
| 4 GiB | 787.0 | 826.2 | 853.1 |
| 16 GiB | 797.2 | 836.2 | 863.5 |

### alltoall

| Message size | node5500 | node5501 | node5502 |
|-------------:|------:|------:|------:|
| 1 MiB | 15.3 | 14.2 | 16.1 |
| 4 MiB | 56.5 | 54.8 | 59.3 |
| 16 MiB | 220.8 | 207.1 | 231.8 |
| 64 MiB | 407.9 | 420.3 | 434.6 |
| 256 MiB | 522.0 | 531.2 | 548.2 |
| 1 GiB | 592.6 | 603.0 | 623.4 |
| 4 GiB | 634.5 | 646.5 | 665.2 |
| 16 GiB | 647.7 | 659.8 | 678.9 |

### hypercube

_No data (run failed or produced no rows)._

