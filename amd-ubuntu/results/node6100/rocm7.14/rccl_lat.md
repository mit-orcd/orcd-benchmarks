# RCCL latency, node6100 + node6101

rccl-tests, 1000 iterations + 100 warm-up per size, no data check, NUMA-bound ranks, 2026-10-13. Time per call in µs (out-of-place). "1 node" columns are XGMI only (no network) for reference; the difference to the 2-node columns is what the network adds.

Runs: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm7.14/rccl/lat_2node_20261013_014214`

## sendrecv

| bytes | 2 nodes, 1 GPU/node | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 1 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 27.4 | 20.7 | 6.2 | 30.8 | 25.8 | 6.3 |
| 64 | 27.1 | 20.5 | 6.2 | 30.6 | 25.7 | 6.2 |
| 1K | 27.8 | 20.4 | 6.4 | 30.8 | 25.7 | 6.4 |
| 8K | 28.1 | 20.6 | 8.6 | 31.1 | 26.6 | 8.6 |
| 64K | 37.0 | 23.2 | 13.3 | 41.0 | 29.2 | 13.3 |
| 1M | 63.1 | 49.1 | 29.6 | 69.5 | 54.9 | 29.6 |
| 4M | 135.5 | 142.5 | 91.3 | 143.7 | 118.5 | 91.3 |

## all_reduce

| bytes | 2 nodes, 1 GPU/node | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 1 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 20.4 | 27.9 | 14.3 | 20.2 | 28.7 | 13.7 |
| 64 | 20.0 | 27.9 | 5.0 | 21.1 | 28.9 | 4.9 |
| 1K | 21.0 | 30.6 | 5.0 | 21.7 | 31.2 | 4.9 |
| 8K | 27.8 | 34.2 | 6.1 | 27.9 | 35.2 | 6.0 |
| 64K | 36.9 | 41.9 | 6.5 | 38.2 | 43.3 | 6.5 |
| 1M | 67.7 | 66.5 | 17.3 | 81.0 | 67.0 | 17.4 |
| 4M | 153.2 | 91.0 | 35.1 | 166.6 | 92.2 | 35.1 |

## all_gather

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|
| 8 | – | – | – | – |
| 64 | – | – | – | – |
| 1K | 36.1 | 7.6 | 38.8 | 7.8 |
| 8K | 36.4 | 7.7 | 39.0 | 7.8 |
| 64K | 38.2 | 8.8 | 41.2 | 8.9 |
| 1M | 54.5 | 11.7 | 76.2 | 11.8 |
| 4M | 73.0 | 27.4 | 73.0 | 27.2 |

## reduce_scatter

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|
| 8 | – | – | – | – |
| 64 | – | – | – | – |
| 1K | 35.6 | 6.0 | 38.6 | 6.0 |
| 8K | 35.9 | 6.1 | 39.2 | 6.1 |
| 64K | 43.3 | 7.1 | 44.1 | 7.2 |
| 1M | 55.0 | 8.7 | 72.0 | 8.7 |
| 4M | 72.4 | 15.9 | 213.8 | 16.1 |

## broadcast

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|
| 8 | 15.9 | 5.9 | 24.6 | 6.6 |
| 64 | 9.6 | 5.8 | 13.5 | 5.7 |
| 1K | 10.1 | 5.8 | 13.8 | 5.7 |
| 8K | 12.8 | 7.8 | 16.2 | 7.7 |
| 64K | 45.5 | 14.9 | 45.7 | 15.1 |
| 1M | 60.1 | 41.7 | 60.7 | 41.7 |
| 4M | 78.5 | 72.6 | 79.3 | 72.5 |

## alltoall

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|
| 8 | – | – | – | – |
| 64 | – | – | – | – |
| 1K | 36.9 | 7.5 | 38.8 | 7.6 |
| 8K | 35.5 | 9.6 | 38.8 | 9.9 |
| 64K | 36.6 | 10.4 | 40.1 | 10.5 |
| 1M | 46.6 | 14.0 | 52.9 | 14.1 |
| 4M | 103.2 | 30.4 | 105.9 | 31.3 |
