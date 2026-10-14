# nccl-tests

## Introduction

[NCCL Tests](https://github.com/NVIDIA/nccl-tests) measure the
performance and correctness of NCCL collective/point-to-point operations
on NVIDIA GPUs. This setup focuses on `sendrecv_perf` (and the other
`*_perf` binaries), reporting in-place/out-of-place bus bandwidth
(busbw, GB/s) for intra-node (NVLink) and inter-node (InfiniBand) GPU
communication.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/nccl-tests/run
./run.sh       "3506 3507" mit_normal_gpu none unlimited 8 l40s 4   # 1 node, all 4 GPUs, per node
./run-2node.sh "3506 3507" mit_normal_gpu none unlimited 8 l40s 4   # every node pair, 1 GPU per node
# when done:
./get-results.sh mit_normal_gpu 2          # sendrecv_perf row at 4 GiB; busbw = 8th column (GB/s)
```

## Installation

See [install.md](install.md): `make MPI=1` against `nvhpc/24.5` + `openmpi/5.0.8`
into `build-nvhpc-24.5-ompi-5.0.8/` (used by `run/`), or `build-nvhpc-26.1.sh`
for B200.

## Usage

### Automated, many runs — `run/`

`run.sh` submits a single-node (all-GPU) job per node; `run-2node.sh`
submits a 2-GPU job for every pair of nodes.

```bash
cd run
# args: "<nodes>" <partition> <reservation|none> <qos> <cpu_count> <gpu_type> <ngpus>
./run.sh       "3506 3507" mit_normal_gpu none unlimited 8 l40s 4
./run-2node.sh "3506 3507" mit_normal_gpu none unlimited 8 l40s 4
```

`run.sh` submits `1node.sh`: one MPI rank drives all `ngpus` GPUs of the node
(`-g ngpus`, NVLink/PCIe path). `run-2node.sh` submits `2nodes-2gpus.sh`: one
rank and one GPU on each node (the InfiniBand path). Both run `sendrecv_perf`
from 1 MiB to 16 GiB (factor 4), using the build that `run/env.sh` selects. The
output goes to `<partition>/out-1node/nccl-1node-<node>-<jobid>` and
`<partition>/out-2node/nccl-2node-2gpu-<node>-<jobid>`.

### Single run — `scripts/` or root

`1node_example.sh` and `2node_example.sh` (repo root) and the scripts in
`scripts/` (`1node.sh`, `2nodes-2gpus.sh`, …) are standalone sbatch
scripts for one configuration. Inside a GPU allocation you can also run
a binary directly:

```bash
cd run && source env.sh                       # sets BUILD_DIR, loads nvhpc/24.5 + openmpi/5.0.8
mpirun -np 1 $BUILD_DIR/sendrecv_perf -b 1M -e 16G -f 4 -g 4      # 1 rank, 4 GPUs
```

## Analysis

Each `*_perf` run prints one table row per message size, plus an average:

```
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
  4294967296    1073741824     float     sum      -1   287491   14.94   14.94      0   286933   14.97   14.97    N/A
# Avg bus bandwidth    : 13.848
```

The first `time / algbw / busbw / #wrong` group is out-of-place and the second
is in-place.

```bash
cd run
./get-results.sh <partition> <N>       # 4 GiB row of the newest N files in out-1node/ and out-2node/
grep -H 'Avg bus bandwidth' ../<partition>/out-*/*      # one number per run
```

Use **busbw** (GB/s) at large sizes (≥ 1 GiB), where it has levelled off. It is
normalised so it can be compared with the link speed. For `sendrecv`,
busbw = algbw. Typical values:

- 1 node: the GPU-to-GPU path. PCIe L40S nodes are about 14 GB/s; NVLink nodes
  are much higher.
- 2 nodes: about 15–23 GB/s with one GPU per node. Values near 2 GB/s mean
  NCCL fell back to TCP sockets (wrong MPI/NCCL stack, or IB not visible).

`#wrong` must be `0` (or `N/A` for in-place `sendrecv`). See `doc/PERFORMANCE.md`
for how busbw is defined.
