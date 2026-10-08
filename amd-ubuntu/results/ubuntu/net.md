# amd-ubuntu — Network: RDMA per ionic rail

Generated 2026-10-08 21:41 by `report.py`.

System: node6100 and node6101, each 8 × AMD Instinct MI355X (gfx950), 2 × EPYC 9575F, 2.2 TiB RAM, Ubuntu 24.04.5, amdgpu 6.19.14, host ROCm 7.2.4 for RVS (ROCm Validation Suite) and rccl-tests, containers under apptainer (Primus, Megatron-LM, ATOM). Nodes linked by 8 × 400G AMD Pollara (ionic) RoCEv2 rails.


## Analysis

- **All 16 rails (8 per node) run at ≈392 Gb/s, 98% of the 400 Gb/s line rate**, alone and with all 8 at once: ≈391–392 GB/s per node.
- **NUMA binding is required.** Without it, all 8 rails at once give only 208 GB/s (52%): rails 0–3 hang off NUMA node 0 and 4–7 off NUMA node 1, and traffic crossing sockets halves the throughput. All multi-node jobs (RCCL, MPI) should bind each rank to its NIC's NUMA node; the 2-node RCCL scripts do this.

## Results

Host-memory RDMA write bandwidth per ionic rail (perftest `ib_write_bw`), alone and all 8 at once. Line rate is 400 Gb/s per rail, 3,200 Gb/s (400 GB/s) per node. Rails 0-3 hang off NUMA node 0 and 4-7 off NUMA node 1; each process must be bound to its NIC's NUMA node to reach line rate on all 8 at once.

### `node6100/ib_bw.md`

perftest `ib_write_bw -x 1 -F --report_gbits -D 10 -s 1048576 -q 4`, NUMA binding=1, 2026-10-01T18:32:35+00:00. Line rate 400 Gb/s per rail, 3,200 Gb/s (400 GB/s) per node.

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

### `node6100/ib_bw_nonuma.md`

perftest `ib_write_bw -x 1 -F --report_gbits -D 10 -s 1048576 -q 4`, **no NUMA binding** (first run, kept for contrast with `ib_bw.md`), 2026-10-01T18:29:52+00:00. Line rate 400 Gb/s per rail, 3,200 Gb/s (400 GB/s) per node.

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

### `node6101/ib_bw.md`

perftest `ib_write_bw -x 1 -F --report_gbits -D 10 -s 1048576 -q 4`, NUMA binding=1, 2026-10-01T18:35:50+00:00. Line rate 400 Gb/s per rail, 3,200 Gb/s (400 GB/s) per node.

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

