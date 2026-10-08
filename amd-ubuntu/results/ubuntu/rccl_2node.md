# amd-ubuntu — RCCL across two nodes

Generated 2026-10-08 15:03 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 for RVS (ROCm Validation Suite) and rccl-tests, containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **With 8 GPUs per node, inter-node collectives reach 92–95% of the 400 GB/s network limit**: all_reduce 381 GB/s, reduce_scatter 379, all_gather 377, broadcast 370 busbw. That is close to the single-node XGMI rate (≈390), so the network is not a bottleneck for data-parallel training across these two nodes.
- RCCL uses the IB/RoCE transport with GPU-direct RDMA on all 8 ionic rails (checked in the probe log).
- With 1–2 GPUs per node busbw stays at one rail (≈48.7 GB/s), and 4 GPUs per node reach ≈150–160 GB/s: each GPU uses its own rail, as expected.
- alltoall (90 GB/s, 22%) is much lower, which is typical for this pattern across nodes; MoE expert-parallel traffic across nodes will be limited by it.
- **sendrecv (29 GB/s, 57% of one rail) is an RCCL default, fixed by one setting** (tested 2026-10-08, [sendrecv-check.md](sendrecv-check.md)): with `NCCL_NCHANNELS_PER_NET_PEER=4` (or 8) it reaches **46.9 GB/s = 94% of one rail**. More queue pairs, a larger chunk size or PXN did not help; raw RDMA reaches 98% per rail. Set it for point-to-point heavy work across nodes (pipeline parallelism, KV-cache transfer); its effect on the other collectives was not tested.

## Results

RCCL across node6100 + node6101 over the 8 ionic rails. Inter-node line rate is 400 GB/s per node; single-node N=8 busbw (XGMI) for scale.

| collective | PPN | ranks | busbw @max (GB/s) | % of line rate | single node N=8 | status |
|---|---:|---:|---:|---:|---:|---|
| all_gather | 1 | 2 | 48.3 | 12% | 388.4 | ok |
| all_gather | 2 | 4 | 48.0 | 12% | 388.4 | ok |
| all_gather | 4 | 8 | 148.1 | 37% | 388.4 | ok |
| all_gather | 8 | 16 | 376.7 | 94% | 388.4 | ok |
| all_reduce | 1 | 2 | 48.7 | 12% | 394.4 | ok |
| all_reduce | 2 | 4 | 48.7 | 12% | 394.4 | ok |
| all_reduce | 4 | 8 | 159.8 | 40% | 394.4 | ok |
| all_reduce | 8 | 16 | 380.9 | 95% | 394.4 | ok |
| alltoall | 8 | 16 | 89.6 | 22% | 343.1 | ok |
| broadcast | 8 | 16 | 369.9 | 92% | 386.7 | ok |
| reduce_scatter | 8 | 16 | 378.5 | 95% | 386.3 | ok |
| sendrecv | 8 | 16 | 28.6 | 7% | 60.7 | ok |
