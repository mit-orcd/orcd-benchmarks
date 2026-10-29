#### 5.1.1 Why the 1-node numbers changed

Turning DDA off does not bring back 7.14's behaviour. With `RCCL_DDA_ENABLE=0`, 12 of 21 points are more than 10% slower than both 7.14 and 10.1's default, in all_reduce, all_gather, reduce_scatter, broadcast, alltoall. So the switch removes DDA and leaves a slower fallback path, not 7.14's path, and this test cannot measure how much of the change DDA explains. 10.1 runs all_reduce, all_gather through the new DDA-IPC implementation (debug log, §2); the DDA switch also changes reduce_scatter, broadcast, alltoall, which log no DDA line. The faster points come with this rework of RCCL's single-node path in 2.30.7 (all_reduce 1M, all_gather 1K, all_gather 64K, all_gather 1M, alltoall 1K, alltoall 64K, alltoall 1M). sendrecv 1K, broadcast 1K change by the same amount with DDA on or off, so that change is separate from DDA (another change in RCCL 2.30.7 / HIP 7.16). Details, same-day timing and RCCL's choices: [`rccl-lat-1node-rocm10.1-why.md`](rccl-lat-1node-rocm10.1-why.md).

| Collective | Size | 7.14 (µs) | 10.1 (µs) | 10.1, DDA off (µs) |
|---|---:|---:|---:|---:|
| sendrecv | 1K | 6.5 | 8.4 | 8.5 |
| all_reduce | 1M | 17.4 | 13.7 | 35.1 |
| all_gather | 1K | 7.8 | 5.0 | 13.4 |
| all_gather | 64K | 9.0 | 6.3 | 14.7 |
| all_gather | 1M | 11.7 | 7.9 | 30.5 |
| broadcast | 1K | 5.8 | 7.7 | 8.0 |
| alltoall | 1K | 7.5 | 4.9 | 9.2 |
| alltoall | 64K | 10.4 | 7.6 | 11.2 |
| alltoall | 1M | 14.0 | 9.2 | 15.7 |
