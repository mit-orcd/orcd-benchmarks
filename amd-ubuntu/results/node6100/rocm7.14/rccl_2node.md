# RCCL 2-node (node6100 + node6101, 8x MI355X each)

Source: /orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm7.14/rccl/rccl_2node_all_20261005_025916, /orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm7.14/rccl/rccl_2node_ppn_20261005_030221

Transport / NICs seen by RCCL: **IB-CAST, ionic_0, ionic_1, ionic_2, ionic_3, ionic_4, ionic_5, ionic_6, ionic_7**

In-place bus bandwidth (GB/s). `lat` is the in-place time of the smallest message (µs).

| collective | PPN | ranks | lat (µs) | 1M | 64M | 1G | 16G | peak busbw | @size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 1 | 2 | 0.1 | 12.0 | 39.5 | 40.3 | 40.4 | **40.4** | 2G |
| all_reduce | 1 | 2 | 21.7 | 15.5 | 43.9 | 47.3 | 48.5 | **48.5** | 16G |
| all_gather | 2 | 4 | 0.1 | 13.6 | 45.9 | 48.2 | 48.9 | **48.9** | 16G |
| all_reduce | 2 | 4 | 20.8 | 12.7 | 47.0 | 48.5 | 48.9 | **48.9** | 16G |
| all_gather | 4 | 8 | 0.1 | 8.9 | 164.3 | 157.8 | 175.5 | **175.5** | 16G |
| all_reduce | 4 | 8 | 24.3 | 21.9 | 176.6 | 165.4 | 177.4 | **177.4** | 16G |
| all_gather | 8 | 16 | 0.1 | 17.3 | 257.8 | 380.0 | 372.3 | **380.0** | 1G |
| all_reduce | 8 | 16 | 29.6 | 28.8 | 211.9 | 384.6 | 372.6 | **384.6** | 1G |
| alltoall | 8 | 16 | 0.1 | 20.0 | 77.5 | 77.4 | - | **77.8** | 128M |
| broadcast | 8 | 16 | 14.1 | 17.6 | 200.6 | 331.6 | 364.6 | **364.6** | 16G |
| reduce_scatter | 8 | 16 | 0.1 | 16.8 | 151.5 | 372.2 | 371.6 | **372.2** | 1G |
| sendrecv | 8 | 16 | 19.3 | 19.9 | 31.6 | 32.0 | 32.1 | **32.1** | 16G |

Reference: 8 × 400 Gb/s NICs per node = 400 GB/s/node unidirectional line rate. For all_reduce, busbw is directly comparable to that per-node link rate.
