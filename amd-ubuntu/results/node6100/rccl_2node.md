# RCCL 2-node (node6100 + node6101, 8x MI355X each)

Source: /orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rccl/rccl_2node_all_20261005_025019, /orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rccl/rccl_2node_ppn_20261005_025420

Transport / NICs seen by RCCL: **IB, ionic_0, ionic_1, ionic_2, ionic_3, ionic_4, ionic_5, ionic_6, ionic_7**

In-place bus bandwidth (GB/s). `lat` is the in-place time of the smallest message (µs).

| collective | PPN | ranks | lat (µs) | 1M | 64M | 1G | 16G | peak busbw | @size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 1 | 2 | 0.1 | 10.8 | 40.6 | 47.6 | 48.3 | **48.3** | 16G |
| all_reduce | 1 | 2 | 22.6 | 12.0 | 44.5 | 48.1 | 48.7 | **48.7** | 16G |
| all_gather | 2 | 4 | 0.1 | 10.8 | 45.8 | 47.9 | 48.0 | **48.0** | 16G |
| all_reduce | 2 | 4 | 22.7 | 14.8 | 47.3 | 48.4 | 48.7 | **48.7** | 16G |
| all_gather | 4 | 8 | 0.1 | 17.0 | 145.2 | 137.6 | 148.1 | **148.1** | 16G |
| all_reduce | 4 | 8 | 25.2 | 20.1 | 164.2 | 152.4 | 159.8 | **164.2** | 64M |
| all_gather | 8 | 16 | 0.1 | 16.8 | 270.4 | 376.8 | 376.7 | **376.8** | 1G |
| all_reduce | 8 | 16 | 31.2 | 32.5 | 301.1 | 378.5 | 380.9 | **380.9** | 16G |
| alltoall | 8 | 16 | 0.0 | 16.6 | 82.7 | 89.0 | - | **89.6** | 4G |
| broadcast | 8 | 16 | 19.2 | 19.1 | 224.9 | 330.1 | 369.9 | **369.9** | 16G |
| reduce_scatter | 8 | 16 | 0.1 | 15.0 | 277.6 | 372.5 | 378.5 | **378.5** | 16G |
| sendrecv | 8 | 16 | 24.7 | 18.8 | 28.7 | 29.2 | 28.6 | **29.2** | 2G |

Reference: 8 × 400 Gb/s NICs per node = 400 GB/s/node unidirectional line rate. For all_reduce, busbw is directly comparable to that per-node link rate.
