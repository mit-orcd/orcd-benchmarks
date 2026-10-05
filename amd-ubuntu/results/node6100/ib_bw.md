# RDMA write bandwidth, node6100 → node6101 (ionic RoCEv2)

perftest `ib_write_bw -x 1 -F --report_gbits -D 10 -s 1048576 -q 4`, NUMA binding=1, 2026-10-01T18:32:35+00:00. Line rate 400 Gb/s per rail, 3,200 Gb/s (400 GB/s) per node.
Raw logs: `logs/node6100/net/ib_bw_20261001_183028`.

| rail | alone (Gb/s) | all 8 at once (Gb/s) | % of 400 alone |
|---|---:|---:|---:|
| ionic_0 | 392.01 | 387.53 | 98% |
| ionic_1 | 392.05 | 390.88 | 98% |
| ionic_2 | 392.14 | 390.36 | 98% |
| ionic_3 | 392.11 | 391.94 | 98% |
| ionic_4 | 391.96 | 391.78 | 98% |
| ionic_5 | 392.13 | 392.07 | 98% |
| ionic_6 | 392.06 | 392.00 | 98% |
| ionic_7 | 392.12 | 392.05 | 98% |
| **total, 8 rails** | | **3128.6** (391.1 GB/s) | 98% of 3,200 |
