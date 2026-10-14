# dgx-nodes

## Introduction

NCCL collective benchmarks on the **DGX H100** nodes running **Ubuntu 24.04**:
node1800/1801, node2700/2701, node2800/2801 (partition `mit_testing`) and
node1700/1701 (partition `ou_orcd_everything`). Each node has 8 × H100 SXM on
NVLink 4 / NVSwitch and NDR 400 Gb/s InfiniBand.

- **1-node:** 8 GPUs in one process, NVLink, 1 MiB → 16 GiB.
- **2-node:** one rank per node (1 GPU/node by default), InfiniBand, within each
  node pair only (no cross-pair runs).
- **Collectives:** all 10 nccl-tests binaries by default.

This is the same test as `../nccl-tests`, but these nodes run Ubuntu, so they
use a separate build (`nccl-tests/build-songhan-utuntu-nvhpc-26.1`: NVHPC 26.1,
CUDA 12.9, NCCL 2.29.2, sm_90) instead of the Rocky 8 build that
`nccl-tests/run/env.sh` selects.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/dgx-nodes
sbatch -p mit_testing -w node1800 --gres=gpu:h100:8 -o out-1node/%x-%N-%J job-nccl-1node.sh 8 sendrecv
sbatch -p mit_testing -w node1800,node1801 --gpus-per-node=h100:1 -o out-2node/%x-%N-%J job-nccl-2node.sh 1 sendrecv
# when done:
./get-results-nccl.sh 2          # sendrecv busbw at 4 GB
./gen-summary.py                 # -> summary.md (peak busbw per collective, vs NVLink / IB ceilings)
```

## Installation

See [install.md](install.md): the NVHPC 26.1 SDK under `../nvhpc` (shared with
`b200-ubuntu`) and `../nccl-tests/build-songhan-utuntu-nvhpc-26.1.sh`, built on an
Ubuntu DGX node. `env-ubuntu.sh` sets the run-time paths.

## Usage

### Automated, many runs — `run-nccl.sh`

Set the variables at the top (`groups` of node pairs, `partition`, `qos`,
`gpus`, `gpus_2node`, `collectives`, `run_1node`, `run_2node`), then:

```bash
./run-nccl.sh          # node180x, 270x, 280x on mit_testing
./run-nccl-1700.sh     # node1700/1701 on ou_orcd_everything
```

Each submits `job-nccl-1node.sh` on every node and `job-nccl-2node.sh` on every
pair. Output goes to `out-1node/` and `out-2node/`.

`watch-and-summarize.sh` (and `-1700.sh`) start `make-summary.sh` in the
background, which waits for the jobs and then rewrites `summary.md`, so the run
can be left after logout.

### Single runs

```bash
sbatch -p mit_testing -w node1800 --gres=gpu:h100:8 -o out-1node/%x-%N-%J job-nccl-1node.sh 8 all
sbatch -p mit_testing -w node1800,node1801 --gpus-per-node=h100:1 -o out-2node/%x-%N-%J job-nccl-2node.sh 1 sendrecv
```

`ib-check/job-ibtest.sh` checks that NCCL uses InfiniBand between two nodes
(`NCCL_DEBUG=INFO`, looks for `NET/IB`). `ib-check/job-tune.sh` compares rank and
GPU-binding layouts for 1-node all_reduce.

## Analysis

```bash
./gen-summary.py            # runs analyze-nccl.py over out-1node/ and out-2node/ -> summary.md
./analyze-nccl.py [json]              # peak busbw + sendrecv/all_reduce sweep tables, to stdout
./get-results-nccl.sh [n_jobs]        # sendrecv at 4 GB, same format as nccl-tests/run/get-results.sh
```

`summary.md` has the peak bus bandwidth (GB/s) per collective and node or pair,
and compares it with the hardware ceilings: NVLink 4 ≈ 478 GB/s per GPU per
direction, one NDR rail ≈ 50 GB/s. `summary-v1.md` is the earlier version, and
`results-oldbuild-tcp/summary-oldbuild-tcp.md` is the first build, where the IB
devices could not be opened and 2-node NCCL fell back to TCP over Ethernet.
