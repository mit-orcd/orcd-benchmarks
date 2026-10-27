# RCCL 2-node (node6100 + node6101, 8x MI355X each)

Source: /orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm10.1/rccl/rccl_2node_all_20261027_181232, /orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm10.1/rccl/rccl_2node_ppn_20261027_181651

Transport / NICs seen by RCCL: **IB-CAST, ionic_0, ionic_1, ionic_2, ionic_3, ionic_4, ionic_5, ionic_6, ionic_7**

In-place bus bandwidth (GB/s). `lat` is the in-place time of the smallest message (µs).

| collective | PPN | ranks | lat (µs) | 1M | 64M | 1G | 16G | peak busbw | @size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 1 | 2 | 0.1 | 12.1 | 39.4 | 40.2 | 41.9 | **41.9** | 16G |
| all_reduce | 1 | 2 | 22.4 | 15.4 | 43.8 | 47.4 | 48.6 | **48.6** | 16G |
| all_gather | 2 | 4 | 0.1 | 13.8 | 45.8 | 48.2 | 48.9 | **48.9** | 16G |
| all_reduce | 2 | 4 | 1419.2 | 12.6 | 47.1 | 48.5 | 48.9 | **48.9** | 16G |
| all_gather | 4 | 8 | 0.1 | 9.0 | 164.0 | 159.0 | 174.8 | **174.8** | 16G |
| all_reduce | 4 | 8 | 1415.5 | 21.8 | 177.8 | 165.7 | 178.3 | **178.3** | 16G |
| all_gather | 8 | 16 | 0.1 | 17.4 | 260.0 | 375.5 | 366.0 | **376.8** | 512M |
| all_reduce | 8 | 16 | 1421.0 | 33.8 | 212.6 | 381.0 | 366.4 | **382.6** | 512M |
| alltoall | 8 | 16 | 0.1 | 21.0 | 77.1 | 76.7 | - | **77.2** | 128M |
| broadcast | 8 | 16 | 14.9 | 19.4 | 232.7 | 326.1 | 360.0 | **360.0** | 16G |
| reduce_scatter | 8 | 16 | 0.1 | 17.2 | 246.6 | 371.5 | 365.1 | **372.4** | 512M |
| sendrecv | 8 | 16 | 18.7 | 20.8 | 31.9 | 32.2 | 32.4 | **32.4** | 16G |

Reference: 8 × 400 Gb/s NICs per node = 400 GB/s/node unidirectional line rate. For all_reduce, busbw is directly comparable to that per-node link rate.
