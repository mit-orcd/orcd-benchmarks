# amd-ubuntu — RCCL collectives, single node (XGMI)

Generated 2026-10-07 20:37 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 for RVS (ROCm Validation Suite) and rccl-tests, containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **N=8 is at the expected XGMI level**: all_reduce 393–394 GB/s, all_gather 388, reduce_scatter 386, alltoall 343 busbw; the two nodes agree within 1%.
- **N=2..4 scale as expected** (one XGMI link ≈ 60 GB/s per pair).
- **N=5..7 dip** (all_reduce ≈ 47 GB/s, 72% below N=4/N=8): a known RCCL topology effect for partial meshes on MI355X, identical on both nodes. Jobs should use 1, 2, 4 or 8 GPUs per node for collective-heavy work.
- sendrecv ≈ 60 GB/s at every N (one point-to-point link), as expected.
- Config sweep (ring, tree, simple protocol, MSCCL off) changes N=8 results by ≤1%; the defaults are fine.

## Results

busbw (GB/s) at the largest message size, XGMI inside one node.

### node6100: busbw vs N

| collective | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 59.6 | 95.8 | 164.7 | 46.0 | 44.8 | 44.6 | 388.4 |
| all_reduce | 59.6 | 93.7 | 168.3 | 47.1 | 46.7 | 46.8 | 394.4 |
| alltoall | 58.5 | 116.1 | 154.8 | 61.0 | 61.8 | 61.6 | 343.1 |
| alltoallv | 58.3 | 46.4 | 114.0 | 22.0 | 27.1 | 46.4 | 216.8 |
| broadcast | 62.0 | 84.9 | 176.0 | 41.9 | 41.8 | 42.0 | 386.7 |
| gather | 61.4 | 122.3 | 182.3 | 70.4 | 74.9 | 89.7 | 425.4 |
| reduce | 61.7 | 99.6 | 168.4 | 48.6 | 48.8 | 48.8 | 327.4 |
| reduce_scatter | 55.8 | 81.7 | 164.6 | 47.6 | 48.2 | 48.6 | 386.3 |
| scatter | 61.6 | 121.0 | 179.9 | 71.1 | 74.4 | 74.0 | 397.1 |
| sendrecv | 58.6 | 61.2 | 60.8 | 61.0 | 60.7 | 60.7 | 60.7 |

Largest N=5..7 dip below min(N=4, N=8): broadcast 76%, all_gather 72%, all_reduce 72%.

### node6101: busbw vs N

| collective | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 59.8 | 95.9 | 164.5 | 45.9 | 44.9 | 44.6 | 387.7 |
| all_reduce | 59.4 | 92.9 | 168.5 | 47.2 | 46.5 | 46.9 | 393.1 |
| alltoall | 58.0 | 115.6 | 154.8 | 60.8 | 61.8 | 61.5 | 342.6 |
| alltoallv | 58.1 | 48.8 | 114.0 | 23.0 | 26.9 | 46.4 | 217.0 |
| broadcast | 61.8 | 84.5 | 175.3 | 41.7 | 41.8 | 41.8 | 387.1 |
| gather | 61.5 | 123.1 | 182.1 | 70.2 | 74.1 | 89.6 | 424.7 |
| reduce | 61.8 | 100.4 | 167.3 | 49.5 | 49.2 | 49.1 | 329.2 |
| reduce_scatter | 55.8 | 82.3 | 164.6 | 47.9 | 48.1 | 48.7 | 386.6 |
| scatter | 61.3 | 121.4 | 180.3 | 69.8 | 74.2 | 73.2 | 397.1 |
| sendrecv | 58.1 | 60.8 | 60.0 | 60.5 | 60.1 | 60.4 | 60.2 |

Largest N=5..7 dip below min(N=4, N=8): broadcast 76%, all_gather 72%, all_reduce 72%.

### N = 8, node6100 vs node6101

| collective | node6100 | node6101 | 6100/6101 |
|---|---:|---:|---:|
| all_gather | 388.4 | 387.7 | 1.00x |
| all_reduce | 394.4 | 393.1 | 1.00x |
| alltoall | 343.1 | 342.6 | 1.00x |
| alltoallv | 216.8 | 217.0 | 1.00x |
| broadcast | 386.7 | 387.1 | 1.00x |
| gather | 425.4 | 424.7 | 1.00x |
| reduce | 327.4 | 329.2 | 0.99x |
| reduce_scatter | 386.3 | 386.6 | 1.00x |
| scatter | 397.1 | 397.1 | 1.00x |
| sendrecv | 60.7 | 60.2 | 1.01x |

### Config sweep, N = 8

| collective | config | node6100 | node6101 | 6100/6101 |
|---|---|---:|---:|---:|
| all_gather | no_mscll | 386.8 | 387.0 | 1.00x |
| all_gather | proto_simple | 388.8 | 387.3 | 1.00x |
| all_gather | ring | 388.2 | 386.1 | 1.01x |
| all_gather | tree | 387.6 | 386.6 | 1.00x |
| all_reduce | no_mscll | 394.4 | 393.0 | 1.00x |
| all_reduce | proto_simple | 393.7 | 392.7 | 1.00x |
| all_reduce | ring | 394.6 | 392.8 | 1.00x |
| all_reduce | tree | 171.0 | 170.6 | 1.00x |
