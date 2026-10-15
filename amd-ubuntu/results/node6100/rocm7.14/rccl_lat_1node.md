# RCCL latency inside one node, node6100 (MI355X)

rccl-tests, 1000 iterations + 100 warm-up per size, no data check, NUMA-bound ranks, 2026-10-15. Time per call in µs (out-of-place). "1 node" columns are XGMI only (no network) for reference; the difference to the 2-node columns is what the network adds.

Runs: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm7.14/rccl/lat_1node_20261014_222719`

## sendrecv

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6.1 | 6.3 | 6.3 | 6.1 | 6.2 | 6.3 |
| 64 | 6.3 | 6.5 | 6.4 | 6.2 | 6.4 | 6.3 |
| 1K | 6.5 | 6.8 | 6.7 | 6.5 | 6.5 | 6.6 |
| 8K | 8.6 | 8.6 | 8.7 | 8.6 | 8.7 | 8.6 |
| 64K | 13.4 | 13.5 | 13.6 | 13.3 | 13.5 | 13.5 |
| 1M | 29.6 | 29.6 | 29.6 | 29.5 | 29.6 | 29.6 |
| 4M | 91.3 | 91.4 | 91.4 | 91.2 | 91.4 | 91.3 |

## all_reduce

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 14.6 | 13.5 | 12.9 | 13.2 | 14.1 | 13.9 |
| 64 | 4.9 | 5.0 | 5.0 | 4.9 | 5.0 | 5.0 |
| 1K | 5.0 | 5.1 | 5.0 | 5.0 | 4.9 | 4.9 |
| 8K | 6.2 | 6.2 | 6.2 | 6.1 | 6.1 | 6.1 |
| 64K | 6.5 | 6.5 | 6.6 | 6.5 | 6.6 | 6.5 |
| 1M | 17.4 | 17.5 | 17.5 | 17.4 | 17.4 | 17.4 |
| 4M | 35.3 | 35.3 | 35.3 | 35.1 | 35.2 | 35.1 |

## all_gather

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 7.8 | 7.9 | 7.8 | 7.8 | 7.7 | 7.6 |
| 8K | 7.9 | 8.0 | 7.9 | 7.9 | 7.8 | 7.8 |
| 64K | 8.8 | 8.9 | 9.2 | 9.0 | 9.1 | 8.8 |
| 1M | 11.7 | 11.8 | 11.7 | 11.9 | 11.7 | 11.7 |
| 4M | 27.5 | 27.4 | 27.1 | 27.3 | 27.3 | 27.5 |

## reduce_scatter

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 6.0 | 6.0 | 6.0 | 6.1 | 6.0 | 6.1 |
| 8K | 6.0 | 6.0 | 6.2 | 6.1 | 6.1 | 6.0 |
| 64K | 6.9 | 7.0 | 7.1 | 7.2 | 7.1 | 7.1 |
| 1M | 8.7 | 8.8 | 8.7 | 8.7 | 8.7 | 8.7 |
| 4M | 15.9 | 16.1 | 15.8 | 15.9 | 15.9 | 16.0 |

## broadcast

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6.1 | 5.6 | 5.9 | 6.4 | 8.5 | 5.7 |
| 64 | 5.8 | 5.8 | 5.6 | 6.0 | 8.3 | 5.8 |
| 1K | 5.8 | 5.9 | 5.7 | 6.1 | 8.3 | 5.8 |
| 8K | 7.9 | 7.7 | 7.6 | 7.7 | 9.0 | 7.8 |
| 64K | 15.0 | 15.2 | 15.0 | 15.2 | 25.5 | 15.3 |
| 1M | 41.7 | 19.6 | 19.8 | 41.6 | 41.6 | 41.7 |
| 4M | 72.5 | 35.9 | 36.0 | 72.7 | 72.5 | 72.6 |

## alltoall

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 7.4 | 7.5 | 7.5 | 7.6 | 7.6 | 7.8 |
| 8K | 9.7 | 9.8 | 9.5 | 9.8 | 10.0 | 9.9 |
| 64K | 10.4 | 10.5 | 10.3 | 10.3 | 10.5 | 10.5 |
| 1M | 14.0 | 13.9 | 13.9 | 14.1 | 14.0 | 14.0 |
| 4M | 30.2 | 30.2 | 30.2 | 29.9 | 30.5 | 31.0 |
