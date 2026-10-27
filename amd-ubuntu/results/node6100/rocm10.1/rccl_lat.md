# RCCL latency, node6100 + node6101

rccl-tests, 1000 iterations + 100 warm-up per size, no data check, NUMA-bound ranks, 2026-10-27. Time per call in µs (out-of-place). "1 node" columns are XGMI only (no network) for reference; the difference to the 2-node columns is what the network adds.

Runs: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm10.1/rccl/lat_2node_20261027_182120`

## sendrecv

| bytes | 2 nodes, 1 GPU/node | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 1 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 27.0 | 18.6 | 7.0 | 31.8 | 26.0 | 7.1 |
| 64 | 26.9 | 18.6 | 8.3 | 31.7 | 25.9 | 8.5 |
| 1K | 28.5 | 18.9 | 8.4 | 31.8 | 26.0 | 8.5 |
| 8K | 28.8 | 19.5 | 10.5 | 32.1 | 26.0 | 10.4 |
| 64K | 37.7 | 21.9 | 13.9 | 42.1 | 29.1 | 14.0 |
| 1M | 62.9 | 48.5 | 30.1 | 69.1 | 54.5 | 30.1 |
| 4M | 135.3 | 142.0 | 91.9 | 142.0 | 122.3 | 91.8 |

## all_reduce

| bytes | 2 nodes, 1 GPU/node | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 1 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 21.5 | 1418.9 | 16.3 | 20.6 | 1426.7 | 16.3 |
| 64 | 20.5 | 1456.1 | 4.9 | 20.9 | 1465.7 | 4.8 |
| 1K | 21.1 | 1678.3 | 5.1 | 21.4 | 1703.0 | 4.9 |
| 8K | 25.8 | 2159.9 | 6.0 | 27.6 | 2160.5 | 5.9 |
| 64K | 38.2 | 2171.8 | 6.4 | 38.9 | 2170.1 | 6.4 |
| 1M | 68.4 | 57.1 | 13.9 | 80.2 | 58.3 | 13.7 |
| 4M | 154.0 | 75.8 | 27.6 | 167.7 | 77.2 | 27.6 |

## all_gather

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|
| 8 | – | – | – | – |
| 64 | – | – | – | – |
| 1K | 36.7 | 5.0 | 38.6 | 5.0 |
| 8K | 37.0 | 5.1 | 39.0 | 5.0 |
| 64K | 38.5 | 6.4 | 41.6 | 6.3 |
| 1M | 54.9 | 7.8 | 73.0 | 7.9 |
| 4M | 73.0 | 16.0 | 73.1 | 16.1 |

## reduce_scatter

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|
| 8 | – | – | – | – |
| 64 | – | – | – | – |
| 1K | 35.4 | 6.2 | 38.5 | 6.4 |
| 8K | 35.6 | 6.2 | 38.7 | 6.4 |
| 64K | 43.3 | 7.1 | 43.2 | 7.1 |
| 1M | 55.0 | 8.8 | 67.2 | 8.7 |
| 4M | 64.2 | 15.6 | 242.3 | 15.7 |

## broadcast

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|
| 8 | 15.5 | 6.0 | 24.9 | 6.2 |
| 64 | 9.8 | 7.6 | 14.1 | 7.6 |
| 1K | 10.1 | 7.7 | 14.5 | 7.6 |
| 8K | 12.3 | 9.1 | 17.0 | 9.2 |
| 64K | 45.1 | 15.8 | 45.4 | 15.7 |
| 1M | 61.7 | 42.4 | 62.4 | 42.5 |
| 4M | 80.6 | 73.3 | 82.0 | 73.6 |

## alltoall

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB |
|---:|---:|---:|---:|---:|
| 8 | – | – | – | – |
| 64 | – | – | – | – |
| 1K | 35.2 | 4.9 | 38.6 | 4.8 |
| 8K | 35.8 | 6.8 | 39.0 | 6.8 |
| 64K | 36.2 | 7.6 | 39.0 | 7.6 |
| 1M | 45.3 | 9.2 | 53.9 | 9.2 |
| 4M | 99.3 | 17.3 | 101.1 | 17.2 |
