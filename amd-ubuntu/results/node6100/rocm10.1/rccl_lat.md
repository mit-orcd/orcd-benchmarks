# RCCL latency, node6100 + node6101

rccl-tests, 1000 iterations + 100 warm-up per size, no data check, NUMA-bound ranks, 2026-10-27. Time per call in µs (out-of-place). "1 node" columns are XGMI only (no network) for reference; the difference to the 2-node columns is what the network adds.

Runs: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm10.1/rccl/lat_2node_20261027_215244`

## sendrecv

| bytes | 2 nodes, 1 GPU/node | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 1 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB | 2 nodes, 1 GPU/node, NCCL_PROTO=LL128 | 2 nodes, 8 GPU/node, NCCL_PROTO=LL128 | 1 node, 8 GPU/node, NCCL_PROTO=LL128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 27.1 | 18.7 | 7.0 | 31.9 | 26.1 | 7.6 | 27.4 | 18.8 | 6.8 |
| 64 | 26.9 | 18.6 | 8.4 | 31.7 | 26.1 | 8.6 | 27.2 | 18.7 | 8.6 |
| 1K | 28.6 | 18.9 | 8.4 | 31.8 | 26.2 | 8.7 | 28.6 | 18.9 | 8.4 |
| 8K | 28.9 | 19.5 | 10.5 | 32.1 | 26.1 | 10.6 | 28.9 | 19.6 | 10.3 |
| 64K | 38.0 | 21.9 | 14.0 | 42.0 | 29.4 | 14.1 | 37.6 | 21.9 | 13.8 |
| 1M | 62.5 | 49.5 | 30.1 | 69.2 | 54.3 | 30.1 | 62.9 | 48.4 | 30.0 |
| 4M | 135.2 | 143.0 | 92.0 | 142.4 | 118.7 | 91.9 | 135.3 | 142.1 | 92.0 |

## all_reduce

| bytes | 2 nodes, 1 GPU/node | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 1 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB | 2 nodes, 1 GPU/node, NCCL_PROTO=LL128 | 2 nodes, 8 GPU/node, NCCL_PROTO=LL128 | 1 node, 8 GPU/node, NCCL_PROTO=LL128 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 21.7 | 1419.0 | 16.3 | 20.8 | 1425.9 | 16.1 | 26.5 | 39.5 | 17.9 |
| 64 | 21.1 | 1456.8 | 5.0 | 21.0 | 1466.9 | 4.9 | 26.3 | 39.4 | 4.9 |
| 1K | 21.7 | 1693.7 | 5.0 | 21.6 | 1719.6 | 4.9 | 26.2 | 39.6 | 5.0 |
| 8K | 26.8 | 2158.1 | 6.0 | 27.0 | 2161.0 | 5.9 | 28.6 | 45.1 | 6.0 |
| 64K | 38.7 | 2171.0 | 6.4 | 40.7 | 2170.9 | 6.3 | 36.6 | 45.4 | 6.4 |
| 1M | 67.3 | 57.3 | 14.0 | 81.4 | 58.5 | 13.7 | 54.7 | 57.4 | 13.7 |
| 4M | 154.9 | 76.0 | 27.6 | 167.7 | 77.1 | 27.6 | 134.1 | 76.1 | 27.6 |

## all_gather

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_PROTO=LL128 | 1 node, 8 GPU/node, NCCL_PROTO=LL128 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 36.7 | 5.0 | 38.4 | 5.1 | 45.3 | 5.0 |
| 8K | 37.0 | 5.0 | 38.6 | 5.1 | 45.4 | 4.9 |
| 64K | 38.7 | 6.3 | 40.9 | 6.3 | 45.6 | 6.3 |
| 1M | 54.6 | 7.8 | 73.5 | 7.9 | 58.8 | 7.9 |
| 4M | 73.1 | 16.1 | 73.2 | 16.1 | 72.9 | 16.1 |

## reduce_scatter

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_PROTO=LL128 | 1 node, 8 GPU/node, NCCL_PROTO=LL128 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 35.6 | 6.0 | 38.6 | 6.1 | 46.9 | 6.4 |
| 8K | 36.0 | 6.2 | 39.0 | 6.1 | 49.1 | 6.2 |
| 64K | 43.3 | 7.0 | 42.6 | 7.0 | 51.8 | 7.0 |
| 1M | 54.7 | 8.7 | 69.8 | 8.7 | 60.3 | 8.7 |
| 4M | 64.1 | 15.7 | 227.4 | 15.7 | 63.4 | 15.6 |

## broadcast

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_PROTO=LL128 | 1 node, 8 GPU/node, NCCL_PROTO=LL128 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 15.2 | 6.2 | 24.6 | 6.1 | 12.0 | 8.2 |
| 64 | 9.9 | 7.7 | 13.9 | 8.0 | 11.4 | 8.9 |
| 1K | 10.1 | 7.7 | 14.3 | 8.0 | 11.3 | 9.0 |
| 8K | 12.7 | 9.2 | 16.7 | 9.3 | 12.0 | 9.2 |
| 64K | 45.2 | 15.6 | 45.5 | 15.7 | 45.3 | 14.9 |
| 1M | 61.6 | 42.5 | 63.1 | 42.5 | 61.2 | 21.1 |
| 4M | 80.6 | 73.5 | 82.2 | 73.5 | 80.6 | 34.0 |

## alltoall

| bytes | 2 nodes, 8 GPU/node | 1 node, 8 GPU/node | 2 nodes, 8 GPU/node, NCCL_NET=IB | 1 node, 8 GPU/node, NCCL_NET=IB | 2 nodes, 8 GPU/node, NCCL_PROTO=LL128 | 1 node, 8 GPU/node, NCCL_PROTO=LL128 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 35.2 | 4.7 | 38.7 | 4.8 | 35.3 | 4.9 |
| 8K | 35.5 | 6.7 | 38.9 | 6.6 | 35.6 | 6.7 |
| 64K | 36.4 | 7.5 | 39.2 | 7.6 | 36.4 | 7.7 |
| 1M | 45.3 | 9.2 | 52.4 | 9.1 | 45.6 | 9.2 |
| 4M | 101.3 | 17.4 | 101.7 | 17.3 | 100.6 | 17.3 |
