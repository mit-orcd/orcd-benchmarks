# RCCL latency inside one node, node6101 (MI355X)

rccl-tests, 1000 iterations + 100 warm-up per size, no data check, NUMA-bound ranks, 2026-10-27. Time per call in µs (out-of-place). "1 node" columns are XGMI only (no network) for reference; the difference to the 2-node columns is what the network adds.

Runs: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/rocm10.1/rccl/lat_1node_20261027_213705`

## sendrecv

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 7.0 | 6.9 | 6.9 | 6.9 | 6.9 | 7.2 |
| 64 | 8.4 | 8.3 | 8.6 | 8.3 | 8.3 | 8.5 |
| 1K | 8.4 | 8.3 | 8.7 | 8.4 | 8.4 | 8.5 |
| 8K | 10.4 | 10.3 | 10.4 | 10.3 | 10.3 | 10.3 |
| 64K | 14.0 | 13.8 | 13.9 | 14.1 | 13.9 | 14.0 |
| 1M | 30.1 | 30.0 | 30.0 | 30.0 | 30.0 | 30.1 |
| 4M | 92.0 | 91.8 | 91.7 | 91.7 | 91.6 | 91.8 |

## all_reduce

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 15.3 | 16.1 | 13.6 | 15.3 | 15.8 | 15.7 |
| 64 | 5.0 | 4.9 | 4.9 | 4.8 | 5.0 | 4.9 |
| 1K | 5.1 | 4.9 | 5.0 | 4.9 | 5.0 | 5.0 |
| 8K | 6.1 | 5.9 | 6.0 | 5.9 | 6.0 | 6.0 |
| 64K | 6.4 | 6.3 | 6.4 | 6.4 | 6.4 | 6.3 |
| 1M | 13.8 | 13.8 | 13.8 | 13.6 | 13.8 | 13.7 |
| 4M | 27.6 | 27.6 | 27.6 | 27.6 | 27.6 | 27.5 |

## all_gather

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 5.2 | 5.1 | 5.1 | 5.0 | 5.1 | 5.0 |
| 8K | 5.1 | 5.1 | 5.0 | 5.0 | 5.0 | 5.0 |
| 64K | 6.3 | 6.3 | 6.3 | 6.4 | 6.4 | 6.3 |
| 1M | 7.8 | 7.8 | 7.8 | 7.8 | 7.8 | 7.8 |
| 4M | 16.0 | 16.0 | 16.0 | 16.1 | 16.0 | 16.0 |

## reduce_scatter

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 6.2 | 6.1 | 6.2 | 6.9 | 6.2 | 6.2 |
| 8K | 6.5 | 6.2 | 6.2 | 6.2 | 6.2 | 6.2 |
| 64K | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 |
| 1M | 8.8 | 8.8 | 8.7 | 8.8 | 8.7 | 8.7 |
| 4M | 15.7 | 15.6 | 15.6 | 15.7 | 15.8 | 15.6 |

## broadcast

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6.1 | 6.6 | 6.1 | 6.1 | 9.2 | 6.2 |
| 64 | 7.7 | 7.6 | 7.7 | 7.8 | 9.7 | 7.8 |
| 1K | 7.6 | 7.6 | 7.8 | 7.8 | 9.7 | 8.0 |
| 8K | 9.1 | 8.9 | 9.2 | 9.2 | 10.2 | 9.4 |
| 64K | 15.4 | 15.5 | 15.4 | 15.5 | 26.4 | 15.5 |
| 1M | 42.4 | 20.3 | 20.1 | 42.5 | 42.5 | 42.5 |
| 4M | 73.5 | 35.7 | 35.8 | 73.3 | 73.6 | 73.6 |

## alltoall

| bytes | 1 node, 8 GPU/node, default, run 1 | 1 node, 8 GPU/node, NCCL_PROTO=LL | 1 node, 8 GPU/node, NCCL_ALGO=Ring NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 2 | 1 node, 8 GPU/node, NCCL_ALGO=Tree NCCL_PROTO=LL | 1 node, 8 GPU/node, default, run 3 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | – | – | – | – | – | – |
| 64 | – | – | – | – | – | – |
| 1K | 4.8 | 4.8 | 4.7 | 4.8 | 4.8 | 4.7 |
| 8K | 6.7 | 6.8 | 6.4 | 6.7 | 6.7 | 6.7 |
| 64K | 7.6 | 7.6 | 7.5 | 7.6 | 7.8 | 7.5 |
| 1M | 9.1 | 9.3 | 9.1 | 9.2 | 9.1 | 9.1 |
| 4M | 17.4 | 17.4 | 17.3 | 17.3 | 17.4 | 17.5 |
