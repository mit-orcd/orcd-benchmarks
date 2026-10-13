# RCCL latency, node6100 + node6101

rccl-tests, 1000 iterations + 100 warm-up per size, no data check, NUMA-bound ranks, 2026-10-13. Time per call in µs (out-of-place). "1 node" columns are XGMI only (no network) for reference; the difference to the 2-node columns is what the network adds.

Runs: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rccl/lat_2node_20261013_014209`

## sendrecv

| bytes | 2 nodes, 1 GPU/node | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node |
|---:|---:|---:|---:|
| 8 | 31.3 | 26.0 | 7.4 |
| 64 | 30.9 | 25.8 | 9.9 |
| 1K | 31.1 | 25.9 | 9.5 |
| 8K | 31.5 | 26.8 | 10.8 |
| 64K | 41.8 | 29.3 | 13.1 |
| 1M | 89.5 | 55.7 | 29.8 |
| 4M | 227.1 | 165.1 | 93.0 |

## all_reduce

| bytes | 2 nodes, 1 GPU/node | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node |
|---:|---:|---:|---:|
| 8 | 21.1 | 29.8 | 13.8 |
| 64 | 21.2 | 29.7 | 15.5 |
| 1K | 22.3 | 32.3 | 15.2 |
| 8K | 51.8 | 35.9 | 15.6 |
| 64K | 56.6 | 44.8 | 15.6 |
| 1M | 87.9 | 59.4 | 24.7 |
| 4M | 158.7 | 79.2 | 47.3 |

## all_gather

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node |
|---:|---:|---:|
| 8 | – | – |
| 64 | – | – |
| 1K | 40.8 | 10.3 |
| 8K | 40.7 | 10.4 |
| 64K | 41.3 | 12.0 |
| 1M | 57.0 | 15.7 |
| 4M | 110.6 | 21.7 |

## reduce_scatter

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node |
|---:|---:|---:|
| 8 | – | – |
| 64 | – | – |
| 1K | 39.7 | 14.1 |
| 8K | 40.5 | 14.9 |
| 64K | 43.2 | 15.1 |
| 1M | 72.1 | 15.8 |
| 4M | 188.6 | 31.1 |

## broadcast

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node |
|---:|---:|---:|
| 8 | 25.1 | 6.0 |
| 64 | 14.1 | 8.6 |
| 1K | 14.3 | 9.2 |
| 8K | 16.6 | 10.1 |
| 64K | 45.4 | 15.0 |
| 1M | 64.0 | 54.3 |
| 4M | 86.8 | 62.7 |

## alltoall

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node |
|---:|---:|---:|
| 8 | – | – |
| 64 | – | – |
| 1K | 40.6 | 9.5 |
| 8K | 40.9 | 8.6 |
| 64K | 41.2 | 9.2 |
| 1M | 57.6 | 12.8 |
| 4M | 109.7 | 21.9 |
