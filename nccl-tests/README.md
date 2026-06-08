# nccl-tests

## Introduction

[NCCL Tests](https://github.com/NVIDIA/nccl-tests) measure the
performance and correctness of NCCL collective/point-to-point operations
on NVIDIA GPUs. This setup focuses on `sendrecv_perf` (and the other
`*_perf` binaries), reporting in-place/out-of-place bus bandwidth
(busbw, GB/s) for intra-node (NVLink) and inter-node (InfiniBand) GPU
communication.

## Installation

Cloned from <https://github.com/NVIDIA/nccl-tests>. Build against an
nvhpc + OpenMPI stack:

```bash
module load nvhpc/24.5 openmpi/5.0.8
make MPI=1                       # binaries land in build/
# or use the helper:
./build.sh                       # see also build-nvhpc.sh / build-ompi.sh
```

Prebuilt trees are kept under `build-nvhpc-*/`. `run/env.sh` selects the
module stack and `BUILD_DIR` used at run time.

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

Output lands in `<partition>/out-1node/` and `<partition>/out-2node/`.

### Single run — `scripts/` or root

`1node_example.sh` and `2node_example.sh` (repo root) and the scripts in
`scripts/` (`1node.sh`, `2nodes-2gpus.sh`, …) are standalone sbatch
scripts for one configuration. Inside a GPU allocation you can also run
a binary directly:

```bash
source run/env.sh
mpirun -np 8 build/sendrecv_perf -b 8 -e 8G -f 2 -g 1
```

## Analysis

```bash
cd run
# get-results.sh <partition> <N>
./get-results.sh mit_normal_gpu 2
```

Pulls the `sendrecv_perf` rows at the 4 GiB (4294967296-byte) message
size from the most recent N outputs; read the **busbw** column (GB/s) —
higher is better. See `doc/PERFORMANCE.md` for what busbw means.
