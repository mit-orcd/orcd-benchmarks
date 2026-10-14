# mpi-io

## Introduction

A small MPI-IO write benchmark. Each of the N MPI ranks writes 10 MB to its own
offset of one shared file (`mpi_io_testfile.dat`), using the collective
`MPI_File_write_at_all`. Rank 0 reports the aggregate write bandwidth. A job
sweeps N = 1, 2, 4, … 32 on one node, showing how shared-file write bandwidth
to a parallel filesystem scales with the number of writers. It does not
measure reads.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/mpi-io
sbatch job.sh                               # set DIR= (filesystem under test) first
# when done:
grep -E 'Run with|Aggregate write' out/mpi-io-bw-<node>-<jobid>
```

## Installation

See [install.md](install.md): one `mpicc` build of `mpi-io-bw.c`. The
`openmpi/5.0.6` module in `job.sh` no longer exists, so rebuild with
`openmpi/5.0.8`.

## Usage

### Single run — root `job.sh`

`job.sh` asks for one node with 32 tasks on `mit_normal`. It copies `mpi-io-bw`
to `DIR`, `cd`s there, and runs `mpirun -np n ./mpi-io-bw` for
n = 1, 2, 4, 8, 16, 32. Edit `#SBATCH -n` and the `for n in ...` list together.

```bash
sbatch job.sh                      # -> out/mpi-io-bw-<node>-<jobid>
sbatch -w node1612 job.sh          # a specific node
```

`c7-job.sh` is the CentOS-7 variant. It uses `mpi-io-bw-c7` and
`DIR=$HOME/orcd/scratch/mpi-io`.

There is no `run/` driver. To cover several nodes or filesystems, submit
`job.sh` once per node, or edit `DIR` and submit once per filesystem.

## Analysis

Each rank count prints:

```
====== Run with 8 MPI tasks ======
Total data written: 80.00 MB
Time taken: 0.0947 seconds
Aggregate write bandwidth: 844.91 MB/s
```

```bash
grep -E 'Run with|Aggregate write' out/*      # rank count -> MB/s
```

Bandwidth should rise with N and level off at what one node or the filesystem
can sustain. The total is small (10 MB × N), so short runs are dominated by
latency and client caching. Treat the numbers as a smoke test and compare them
only across runs of the same size. A sudden collapse, e.g. 30 s for 320 MB,
points to lock contention or a filesystem problem. For sustained bandwidth, use
`fio`.
