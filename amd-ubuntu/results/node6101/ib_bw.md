# RDMA write bandwidth, node6101 → node6100 (ionic RoCEv2)

perftest `ib_write_bw -x 1 -F --report_gbits -D 10 -s 1048576 -q 4`, NUMA binding=1, 2026-10-01T18:35:50+00:00. Line rate 400 Gb/s per rail, 3,200 Gb/s (400 GB/s) per node.
Raw logs: `logs/node6101/net/ib_bw_20261001_183342`.

| rail | alone (Gb/s) | all 8 at once (Gb/s) | % of 400 alone |
|---|---:|---:|---:|
| ionic_0 | 392.07 | 392.12 | 98% |
| ionic_1 | 392.11 | 392.12 | 98% |
| ionic_2 | 392.10 | 392.12 | 98% |
| ionic_3 | 391.90 | 392.12 | 98% |
| ionic_4 | 392.12 | 391.91 | 98% |
| ionic_5 | 392.08 | 392.05 | 98% |
| ionic_6 | 391.96 | 392.10 | 98% |
| ionic_7 | 392.04 | 392.10 | 98% |
| **total, 8 rails** | | **3136.6** (392.1 GB/s) | 98% of 3,200 |
