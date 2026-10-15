# RCCL latency inside one node, node6101 (MI355X)

rccl-tests, 1000 iterations + 100 warm-up per size, no data check, NUMA-bound ranks, 2026-10-15. Time per call in µs (out-of-place). "1 node" columns are XGMI only (no network) for reference; the difference to the 2-node columns is what the network adds.

Runs: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/rocm7.14/rccl/lat_1node_20261014_222717`

## sendrecv

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6.5 | 6.2 | 6.3 | 6.2 | 6.3 | 6.0 |
| 64 | 6.2 | 6.2 | 6.1 | 6.3 | 6.3 | 6.2 |
| 1K | 6.4 | 6.5 | 6.4 | 6.5 | 6.5 | 6.4 |
| 8K | 8.6 | 8.6 | 8.6 | 8.6 | 8.6 | 8.6 |
| 64K | 13.4 | 13.3 | 13.3 | 13.5 | 13.4 | 13.3 |
| 1M | 29.5 | 29.5 | 29.5 | 29.6 | 29.5 | 29.5 |
| 4M | 91.2 | 91.3 | 91.2 | 91.2 | 91.2 | 91.2 |

## all_reduce

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 14.7 | 14.2 | 12.4 | 14.1 | 14.9 | 14.6 |
| 64 | 5.0 | 5.0 | 5.0 | 4.9 | 5.0 | 5.0 |
| 1K | 5.0 | 5.0 | 5.1 | 4.9 | 5.1 | 5.1 |
| 8K | 6.1 | 6.1 | 6.2 | 6.1 | 6.2 | 6.1 |
| 64K | 6.6 | 6.5 | 6.5 | 6.5 | 6.5 | 6.5 |
| 1M | 17.3 | 17.4 | 17.5 | 17.4 | 17.4 | 17.4 |
| 4M | 35.1 | 35.4 | 35.3 | 35.3 | 35.4 | 35.3 |

## all_gather

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 7.7 | 7.7 | 7.7 | 7.8 | 7.8 | 7.8 |
| 8K | 7.8 | 7.8 | 7.8 | 7.9 | 7.8 | 8.0 |
| 64K | 9.1 | 8.8 | 8.8 | 8.9 | 9.0 | 9.1 |
| 1M | 11.8 | 11.8 | 11.7 | 11.7 | 11.8 | 11.8 |
| 4M | 27.2 | 27.3 | 27.4 | 27.3 | 26.8 | 27.6 |

## reduce_scatter

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 6.1 | 5.9 | 6.0 | 6.2 | 6.2 | 5.9 |
| 8K | 6.1 | 6.1 | 6.0 | 6.1 | 6.1 | 6.0 |
| 64K | 7.1 | 7.1 | 7.0 | 7.1 | 7.1 | 6.8 |
| 1M | 8.7 | 8.7 | 8.7 | 8.6 | 8.6 | 8.8 |
| 4M | 16.0 | 15.9 | 16.0 | 16.1 | 16.3 | 16.1 |

## broadcast

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5.6 | 5.6 | 5.6 | 5.7 | 8.4 | 5.5 |
| 64 | 5.7 | 5.7 | 5.8 | 5.8 | 8.2 | 5.7 |
| 1K | 5.7 | 5.8 | 5.8 | 5.8 | 8.2 | 5.7 |
| 8K | 7.6 | 7.6 | 7.7 | 7.6 | 8.9 | 7.6 |
| 64K | 15.1 | 15.1 | 15.1 | 15.0 | 25.4 | 14.9 |
| 1M | 41.7 | 19.8 | 19.7 | 41.7 | 41.7 | 41.6 |
| 4M | 72.6 | 35.9 | 35.8 | 72.4 | 72.6 | 72.6 |

## alltoall

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 7.7 | 7.6 | 7.6 | 7.6 | 7.5 | 7.7 |
| 8K | 10.0 | 9.9 | 9.9 | 9.8 | 9.7 | 9.8 |
| 64K | 10.3 | 10.3 | 10.2 | 10.4 | 10.5 | 10.5 |
| 1M | 13.9 | 13.5 | 14.2 | 14.2 | 14.0 | 14.2 |
| 4M | 30.7 | 30.8 | 30.5 | 30.5 | 30.8 | 31.0 |
