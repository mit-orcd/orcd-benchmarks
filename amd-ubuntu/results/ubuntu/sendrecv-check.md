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
