# mpi-io

## Introduction

A parallel I/O benchmark using MPI-IO. Each of the N MPI ranks writes and
reads a shared/striped file via MPI-IO collective calls, and the
aggregate bandwidth (MB/s or GB/s) is reported. It measures how
parallel-filesystem write/read bandwidth scales with the number of MPI
ranks on a node.

## Installation

The source is in this directory; build with an MPI compiler:

```bash
module load openmpi/5.0.6
mpicc -O3 mpi-io-bw.c -o mpi-io-bw
# c7 (CentOS-7) build: inp-mpi-io-bw.c -> mpi-io-bw-c7
```

The job copies the binary to a scratch directory
(`/orcd/scratch/orcd/022/shaohao/mpi-io`) and runs there.

## Usage

### Single run — root `job.sh`

`job.sh` is a one-node sbatch script that runs the benchmark over a rank
sweep (1, 2, 4, 8, 16, 32). Edit the `#SBATCH` partition / `-n` and the
scratch `DIR`, then:

```bash
sbatch job.sh        # -> out/<job>.out
# c7-job.sh is the CentOS-7 variant
```

There is no `run/` many-runs driver for this benchmark; to sweep nodes,
submit `job.sh` to each node (e.g. `sbatch -w <node> job.sh`).

## Analysis

Each `====== Run with <n> MPI tasks ======` block prints the MPI-IO
write/read bandwidth for that rank count (output under `out/`). Compare
bandwidth across rank counts to see parallel-I/O scaling; higher is
better and it typically plateaus at the filesystem's limit.
