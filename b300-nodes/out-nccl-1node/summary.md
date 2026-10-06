# nccl-tests 1-node collective summary

- Generated: 2026-10-05 16:34:41
- Nodes: node5900-c1 (each 8 x NVIDIA B300 SXM6 AC, single node, intra-node NVLink)
- Config: 1 thread, 1 MiB-16 GiB, 5 warmup + 20 iters
- Collectives: sendrecv, reduce, broadcast, gather, scatter, reduce_scatter, all_gather, all_reduce, alltoall, hypercube
- Reference: MIT aicr-benchmarks `results_b200.md`, Table 1 (b0027, 8x B200, NVLink 5.0 / NVSwitch), busbw at 900 GB/s NVLink max

Converged busbw = busbw at the largest message size, best of out-of-place / in-place (matches the reference methodology). busbw (bus bandwidth) is the figure of merit.

## Converged bus bandwidth by collective (GB/s)

| Collective | node5900-c1 | Reference (b0027) | node5900-c1 % of ref | Correctness |
|---|---:|---:|---:|---|
| sendrecv | 638.4 | 666 | 96% | PASS |
| reduce | 657.3 | 701 | 94% | PASS |
| broadcast | 670.5 | 691 | 97% | PASS |
| gather | 797.5 | 717 | 111% | PASS |
| scatter | 689.5 | 746 | 92% | PASS |
| reduce_scatter | 759.9 | 695 | 109% | PASS |
| all_gather | 646.0 | 684 | 94% | PASS |
| all_reduce | 914.3 | 841 | 109% | PASS |
| alltoall | 628.0 | 675 | 93% | PASS |
| hypercube | FAILED | — | — | FAIL |

## Bus bandwidth vs message size (out-of-place busbw, GB/s)

### sendrecv

| Message size | node5900-c1 |
|-------------:|------:|
| 1 MiB | 30.0 |
| 4 MiB | 65.4 |
| 16 MiB | 80.7 |
| 64 MiB | 87.2 |
| 256 MiB | 323.4 |
| 1 GiB | 619.4 |
| 4 GiB | 635.2 |
| 16 GiB | 638.4 |

### reduce

| Message size | node5900-c1 |
|-------------:|------:|
| 1 MiB | 27.0 |
| 4 MiB | 100.8 |
| 16 MiB | 313.6 |
| 64 MiB | 504.8 |
| 256 MiB | 622.7 |
| 1 GiB | 663.1 |
| 4 GiB | 677.1 |
| 16 GiB | 654.1 |

### broadcast

| Message size | node5900-c1 |
|-------------:|------:|
| 1 MiB | 29.6 |
| 4 MiB | 108.2 |
| 16 MiB | 330.0 |
| 64 MiB | 537.9 |
| 256 MiB | 664.4 |
| 1 GiB | 698.4 |
| 4 GiB | 713.1 |
| 16 GiB | 670.5 |

### gather

| Message size | node5900-c1 |
|-------------:|------:|
| 1 MiB | 28.7 |
| 4 MiB | 109.9 |
| 16 MiB | 442.2 |
| 64 MiB | 681.7 |
| 256 MiB | 774.0 |
| 1 GiB | 778.6 |
| 4 GiB | 796.4 |
| 16 GiB | 797.5 |

### scatter

| Message size | node5900-c1 |
|-------------:|------:|
| 1 MiB | 27.1 |
| 4 MiB | 107.0 |
| 16 MiB | 409.0 |
| 64 MiB | 609.2 |
| 256 MiB | 704.8 |
| 1 GiB | 683.0 |
| 4 GiB | 685.9 |
| 16 GiB | 689.5 |

### reduce_scatter

| Message size | node5900-c1 |
|-------------:|------:|
| 1 MiB | 22.5 |
| 4 MiB | 89.9 |
| 16 MiB | 160.1 |
| 64 MiB | 456.4 |
| 256 MiB | 645.6 |
| 1 GiB | 706.3 |
| 4 GiB | 742.6 |
| 16 GiB | 758.9 |

### all_gather

| Message size | node5900-c1 |
|-------------:|------:|
| 1 MiB | 18.6 |
| 4 MiB | 73.5 |
| 16 MiB | 132.0 |
| 64 MiB | 396.9 |
| 256 MiB | 554.8 |
| 1 GiB | 592.3 |
| 4 GiB | 626.5 |
| 16 GiB | 640.6 |

### all_reduce

| Message size | node5900-c1 |
|-------------:|------:|
| 1 MiB | 38.7 |
| 4 MiB | 121.6 |
| 16 MiB | 256.7 |
| 64 MiB | 402.8 |
| 256 MiB | 717.0 |
| 1 GiB | 798.4 |
| 4 GiB | 904.4 |
| 16 GiB | 913.5 |

### alltoall

| Message size | node5900-c1 |
|-------------:|------:|
| 1 MiB | 15.7 |
| 4 MiB | 58.7 |
| 16 MiB | 228.9 |
| 64 MiB | 404.0 |
| 256 MiB | 516.5 |
| 1 GiB | 586.7 |
| 4 GiB | 614.6 |
| 16 GiB | 628.0 |

### hypercube

_No data (run failed or produced no rows)._

