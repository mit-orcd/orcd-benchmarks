#### 5.1.1 Why the 1-node numbers changed

4 of the points that changed by more than 10% are in collectives that take RCCL 10.1's new DDA path (all_reduce, all_gather). With `RCCL_DDA_ENABLE=0`, 0 of these 4 come back to within 10% of 7.14. So the DDA path does not explain the change in these collectives. 7 changed points are in collectives without DDA (sendrecv 8, sendrecv 1K, broadcast 8, broadcast 1K, alltoall 1K, alltoall 64K, alltoall 1M); turning DDA off brings 2 of them back. These come from another change in RCCL 2.30.7 / HIP 7.16 (lists of new debug lines below). Details, same-day timing and RCCL's choices: [`rccl-lat-1node-rocm10.1-why.md`](rccl-lat-1node-rocm10.1-why.md).

| Collective | Size | 7.14 (µs) | 10.1 (µs) | 10.1, DDA off (µs) |
|---|---:|---:|---:|---:|
| sendrecv | 8 | 7.7 | 7.0 | 7.3 |
| sendrecv | 1K | 6.5 | 8.4 | 8.5 |
| all_reduce | 1M | 17.4 | 13.7 | 35.1 |
| all_gather | 1K | 7.8 | 5.0 | 13.4 |
| all_gather | 64K | 9.0 | 6.3 | 14.7 |
| all_gather | 1M | 11.7 | 7.9 | 30.5 |
| broadcast | 8 | 7.0 | 6.1 | 7.8 |
| broadcast | 1K | 5.8 | 7.7 | 8.0 |
| alltoall | 1K | 7.5 | 4.9 | 9.2 |
| alltoall | 64K | 10.4 | 7.6 | 11.2 |
| alltoall | 1M | 14.0 | 9.2 | 15.7 |
