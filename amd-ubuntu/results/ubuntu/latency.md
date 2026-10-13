# Network latency, node6100 ↔ node6101

Updated 2026-10-13 by `rccl-tests/compare_lat.py ubuntu`. Scripts: `net/run_ib_lat.sh` (host RDMA, perftest), `rccl-tests/run_rccl_lat_2node.sh` (RCCL, rccl-tests 8 B–4 MB, 1000 iterations + 100 warm-up, no data check, one rank per GPU, NUMA-bound).

Compared: host ROCm 7.2.4 (RCCL 2.27.7) with ROCm 7.14 (RCCL 2.30.4, default IB-CAST transport and `NCCL_NET=IB`). Time per call in µs, lower is better.

## 1. Raw network: RDMA latency, one rail at a time (no GPU, no RCCL)

perftest one-way latency (median, µs), host memory. Logs: `logs/node6100/net/ib_lat_20261013_014209`.

| test | bytes | median of 8 rails | fastest rail | slowest rail |
|---|---:|---:|---:|---:|
| ib_write_lat | 2 | 5.21 | 5.19 | 5.23 |
| ib_write_lat | 8 | 5.21 | 5.19 | 5.23 |
| ib_write_lat | 64 | 5.23 | 5.21 | 5.25 |
| ib_write_lat | 4096 | 6.34 | 6.32 | 6.34 |
| ib_write_lat | 65536 | 7.97 | 7.92 | 8.00 |
| ib_send_lat | 2 | 6.72 | 6.67 | 6.74 |
| ib_send_lat | 8 | 6.72 | 6.68 | 6.74 |
| ib_send_lat | 64 | 6.74 | 6.69 | 6.75 |
| ib_send_lat | 4096 | 7.91 | 7.84 | 7.95 |
| ib_send_lat | 65536 | 9.28 | 9.26 | 9.31 |

## 2. RCCL, 2 nodes

Columns: 7.2.4 | 7.14 default | 7.14 `NCCL_NET=IB` | 1 node × 8 GPUs on 7.14 (XGMI only, no network). The ratio is 7.2.4 time ÷ 7.14 time (>1: 7.14 faster).

### 2 nodes × 1 GPU (GPU → NIC → wire → NIC → GPU)

| collective | bytes | 7.2.4 | 7.14 default | 7.14 NCCL_NET=IB | 7.14 best vs 7.2.4 | 1 node, 7.14 |
|---|---:|---:|---:|---:|---:|---:|
| sendrecv | 8 | 31.3 | 27.4 | 30.8 | 1.14x | – |
| sendrecv | 1K | 31.1 | 27.8 | 30.8 | 1.12x | – |
| sendrecv | 64K | 41.8 | 37.0 | 41.0 | 1.13x | – |
| sendrecv | 1M | 89.5 | 63.1 | 69.5 | 1.42x | – |
| all_reduce | 8 | 21.1 | 20.4 | 20.2 | 1.04x | – |
| all_reduce | 1K | 22.3 | 21.0 | 21.7 | 1.06x | – |
| all_reduce | 64K | 56.6 | 36.9 | 38.2 | 1.53x | – |
| all_reduce | 1M | 87.9 | 67.7 | 81.0 | 1.30x | – |

### 2 nodes × 8 GPUs (16 ranks, as in the bandwidth tables)

| collective | bytes | 7.2.4 | 7.14 default | 7.14 NCCL_NET=IB | 7.14 best vs 7.2.4 | 1 node, 7.14 |
|---|---:|---:|---:|---:|---:|---:|
| sendrecv | 8 | 26.0 | 20.7 | 25.8 | 1.26x | 6.2 |
| sendrecv | 1K | 25.9 | 20.4 | 25.7 | 1.27x | 6.4 |
| sendrecv | 64K | 29.3 | 23.2 | 29.2 | 1.26x | 13.3 |
| sendrecv | 1M | 55.7 | 49.1 | 54.9 | 1.13x | 29.6 |
| all_reduce | 8 | 29.8 | 27.9 | 28.7 | 1.07x | 14.3 |
| all_reduce | 1K | 32.3 | 30.6 | 31.2 | 1.06x | 5.0 |
| all_reduce | 64K | 44.8 | 41.9 | 43.3 | 1.07x | 6.5 |
| all_reduce | 1M | 59.4 | 66.5 | 67.0 | 0.89x | 17.3 |
| all_gather | 1K | 40.8 | 36.1 | 38.8 | 1.13x | 7.6 |
| all_gather | 64K | 41.3 | 38.2 | 41.2 | 1.08x | 8.8 |
| all_gather | 1M | 57.0 | 54.5 | 76.2 | 1.04x | 11.7 |
| reduce_scatter | 1K | 39.7 | 35.6 | 38.6 | 1.12x | 6.0 |
| reduce_scatter | 64K | 43.2 | 43.3 | 44.1 | 1.00x | 7.1 |
| reduce_scatter | 1M | 72.1 | 55.0 | 72.0 | 1.31x | 8.7 |
| broadcast | 8 | 25.1 | 15.9 | 24.6 | 1.58x | 5.9 |
| broadcast | 1K | 14.3 | 10.1 | 13.8 | 1.42x | 5.8 |
| broadcast | 64K | 45.4 | 45.5 | 45.7 | 1.00x | 14.9 |
| broadcast | 1M | 64.0 | 60.1 | 60.7 | 1.06x | 41.7 |
| alltoall | 1K | 40.6 | 36.9 | 38.8 | 1.10x | 7.5 |
| alltoall | 64K | 41.2 | 36.6 | 40.1 | 1.12x | 10.4 |
| alltoall | 1M | 57.6 | 46.6 | 52.9 | 1.24x | 14.0 |

Runs: 7.2.4 `logs/node6100/rccl/lat_2node_20261013_014209`, 7.14 `logs/node6100/rocm7.14/rccl/lat_2node_20261013_014214`.
