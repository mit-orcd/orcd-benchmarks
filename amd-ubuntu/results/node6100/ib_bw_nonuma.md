# RDMA write bandwidth, node6100 → node6101 (ionic RoCEv2)

perftest `ib_write_bw -x 1 -F --report_gbits -D 10 -s 1048576 -q 4`, **no NUMA binding** (first run, kept for contrast with `ib_bw.md`), 2026-10-01T18:29:52+00:00. Line rate 400 Gb/s per rail, 3,200 Gb/s (400 GB/s) per node.
Raw logs: `logs/node6100/net/ib_bw_20261001_182745`.

| rail | alone (Gb/s) | all 8 at once (Gb/s) | % of 400 alone |
|---|---:|---:|---:|
| ionic_0 | 392.16 | 120.39 | 98% |
| ionic_1 | 392.16 | 145.36 | 98% |
| ionic_2 | 360.77 | 104.72 | 90% |
| ionic_3 | 392.11 | 127.38 | 98% |
| ionic_4 | 392.13 | 293.21 | 98% |
| ionic_5 | 392.16 | 310.97 | 98% |
| ionic_6 | 392.11 | 269.17 | 98% |
| ionic_7 | 392.15 | 293.84 | 98% |
| **total, 8 rails** | | **1665.0** (208.1 GB/s) | 52% of 3,200 |
