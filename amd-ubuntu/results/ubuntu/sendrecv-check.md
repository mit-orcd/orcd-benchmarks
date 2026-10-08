# 2-node sendrecv check (node6100 → node6101)

Raw logs: `logs/node6100/rccl/sendrecv_check_20261008_021512`. Script: `rccl-tests/run_sendrecv_check.sh`.
Question: why is 2-node sendrecv only 57% of one rail (28.6 of 50 GB/s) when the other collectives reach 92–95% of 400 GB/s?

## A. Raw RDMA on one rail (ib_write_bw, host memory, 1 MiB, 10 s)

| path | queue pairs | Gb/s | % of 400 |
|---|---:|---:|---:|
| ionic_0 → ionic_0 (same rail) | 1 | 281 | 70% |
| ionic_0 → ionic_0 (same rail) | 2 | 392 | 98% |
| ionic_0 → ionic_0 (same rail) | 4 | 392 | 98% |
| ionic_7 → ionic_0 (cross rail) | 1 | 267 | 67% |
| ionic_7 → ionic_0 (cross rail) | 4 | 392 | 98% |

## B. RCCL sendrecv with different settings (16 GiB, busbw)

| run | ranks per node | setting | busbw at 16 GiB (GB/s) | best (GB/s) | % of one rail | transport |
|---|---:|---|---:|---:|---:|---|
| default_ppn1 | 1 | `default` | 22.1 | 22.1 | 44% | GDRDMA |
| default | 8 | `default` | 28.5 | 29.1 | 57% | GDRDMA |
| nch4 | 8 | `NCCL_NCHANNELS_PER_NET_PEER=4` | 46.9 | 47.6 | 94% | GDRDMA |
| nch8 | 8 | `NCCL_NCHANNELS_PER_NET_PEER=8` | 46.8 | 47.1 | 94% | GDRDMA |
| qps2 | 8 | `NCCL_IB_QPS_PER_CONNECTION=2` | 28.8 | 29.3 | 58% | GDRDMA |
| qps4 | 8 | `NCCL_IB_QPS_PER_CONNECTION=4` | 28.6 | 29.2 | 57% | GDRDMA |
| chunk1m | 8 | `NCCL_P2P_NET_CHUNKSIZE=1048576` | 28.4 | 28.8 | 57% | GDRDMA |
| pxn | 8 | `NCCL_PXN_DISABLE=0 NCCL_P2P_PXN_LEVEL=2` | 28.7 | 29.4 | 57% | GDRDMA |
| nch8_qps4 | 8 | `NCCL_NCHANNELS_PER_NET_PEER=8 NCCL_IB_QPS_PER_CONNECTION=4` | 46.5 | 47.1 | 93% | GDRDMA |

How to read it:
- **A, 1 queue pair well below 400 Gb/s** → one RDMA connection cannot fill a rail; RCCL settings that add queue pairs or channels (B: qps*, nch*) should then help.
- **A, cross rail slower than same rail** → the 16-rank sendrecv pair GPU7@node6100 → GPU0@node6101 goes from rail 7 to rail 0 and leaves its rail; the `pxn` run (route through the GPU on the matching rail, as NCCL does on B200) should then help.
- **B, `default_ppn1`** (GPU0 ↔ GPU0, same rail) vs `default` (16 ranks, includes the cross-rail pair) separates the two causes.

Baseline `default` run: 28.5 GB/s (57% of one rail).

## Result (2026-10-08)

**Compared: RCCL sendrecv across the two nodes with its default settings vs with one setting changed at a time (ROCm 7.2.4, node6100 → node6101), and raw RDMA on one rail.**

- **The network is fine:** raw RDMA reaches 392 Gb/s = 98% of a rail with 2 or more queue pairs, on the same rail and on crossed rails (rail 7 → rail 0) alike. One queue pair alone gives 67–70%.
- **The fix is `NCCL_NCHANNELS_PER_NET_PEER=4`** (or 8): sendrecv rises from 28.5 to **46.9 GB/s = 94% of one rail**, 1.65× the default and close to B200's 48.8 GB/s (98%).
- **The cause is RCCL's default number of channels per network peer** for point-to-point traffic: each channel moves only part of a rail, and the default opens too few. More queue pairs per connection (`qps2`, `qps4`), a 1 MiB chunk size and PXN routing all stay at 57–58%, so the limit is per channel, not per connection, and rail crossing is not the cause (PXN does not help).
- With one GPU per node (`default_ppn1`, same rail) the default is even lower (44%), consistent with too few channels per peer.
- **Suggestion:** set `NCCL_NCHANNELS_PER_NET_PEER=4` for point-to-point heavy work across nodes (pipeline parallelism, KV-cache transfer, MoE expert parallelism). Its effect on the ring collectives (already 92–95%) was not tested.

## ROCm 7.14: network transport test (2026-10-08)

Compared: ROCm 7.14 on our nodes with different RCCL network settings (2 nodes, 16 GPUs, busbw at the largest size: alltoall 4 GB, sendrecv 16 GiB). ROCm 7.14 ships RCCL 2.30.4, which uses a new network transport (`IB-CAST`); the host ROCm 7.2.4 ships RCCL 2.27.7 with the plain `IB` transport. Logs: `logs/node6100/rocm7.14/rccl/p2p_transport_714_*` and `rccl_2node_p2p714_*`. Script: `rccl-tests/run_p2p_transport_714.sh`.

| setting (ROCm 7.14) | transport | alltoall GB/s (% of 93.75) | sendrecv GB/s (% of 50) |
|---|---|---:|---:|
| default | IB-CAST | 73.7 (79%) | 32.1 (64%) |
| `NCCL_NCHANNELS_PER_NET_PEER=4` | IB-CAST | 56.6 (60%) | 40.1 (80%) |
| `NCCL_NET=IB` | IB | **89.6 (96%)** | 38.5 (77%) |
| `NCCL_NET=IB NCCL_NCHANNELS_PER_NET_PEER=4` | IB | 54.0 (58%) | **49.4 (99%)** |
| `NCCL_IB_QP_SCHED_ENABLE=1` | IB-CAST | 73.6 (79%) | 29.5 (59%) |
| `NCCL_IB_QP_SCHED_ENABLE=1 NCCL_NCHANNELS_PER_NET_PEER=4` | IB-CAST | 52.4 (56%) | 39.8 (80%) |

- **The new `IB-CAST` transport in ROCm 7.14's RCCL is why point-to-point is slower on 7.14 than on 7.2.4.** With `NCCL_NET=IB` (the old transport), ROCm 7.14 gives alltoall 89.6 GB/s, the same as 7.2.4, and sendrecv with 4 channels per peer reaches 49.4 GB/s = 99% of one rail (7.2.4: 46.9; B200: 48.8).
- Turning the queue-pair scheduler back on (`NCCL_IB_QP_SCHED_ENABLE=1`) does not help.
- **`NCCL_NCHANNELS_PER_NET_PEER=4` helps sendrecv but hurts alltoall** (73.7 → 56.6 GB/s with IB-CAST, 89.6 → 54.0 with IB). So it should not be set globally; set it only for jobs dominated by sendrecv-style traffic (e.g. pipeline parallelism).
- Suggested on ROCm 7.14: `NCCL_NET=IB` for all multi-node jobs; add `NCCL_NCHANNELS_PER_NET_PEER=4` only for sendrecv-heavy jobs. The ring collectives (all_reduce etc.) were not rerun with `NCCL_NET=IB`.
- Single runs; the default alltoall here (73.7) is 4% below the earlier 7.14 run (77.1), which gives the run-to-run spread.
