# mpi-laplace

## Introduction

An MPI benchmark that solves the 2-D Laplace equation by Jacobi iteration on a
1000 × 1000 grid until the max change is below 0.01 (3372 iterations). The MPI
version splits the rows over **exactly 4 ranks**, exchanges halo rows with
`MPI_Send` / `MPI_Recv` every iteration, and reduces the error with
`MPI_Allreduce`. The serial version is the baseline. The result is a quick
check of per-node compute plus intra-node MPI performance. `src/` also holds
other small MPI teaching codes (calc-pi, ring, bcast, …) that are not part of
the benchmark.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/mpi-laplace/work
mkdir -p <partition>/output
# set nodes=(...) and partition= at the top of job_all_example.sh, then:
./job_all_example.sh
# when done:
grep 'Total time' <partition>/output/<node>-<jobid>.out     # 1st = serial, 2nd = MPI (4 ranks)
```

## Installation

See [install.md](install.md). Prebuilt binaries are in `src/bin-r8/`; rebuild
with `gcc` and `mpicc` if needed.

## Usage

Both programs read the maximum iteration count (`work/inp` = 4000) from stdin.
`laplace_mpi` must run with 4 ranks, or it exits with
`This code must be run with 4 PEs`.

### Automated, many runs — `work/job_all_example.sh`

`job_all_example.sh` submits one job per node in `nodes=(...)` (8 tasks,
10 min). Each job prints `lscpu`, then runs the serial solver and
`mpirun -np 4 laplace_mpi`. Edit `nodes` and `partition` at the top, then:

```bash
cd work
./job_all_example.sh       # -> work/<partition>/output/<node>-<jobid>.out
```

### Single run — `work/`

```bash
salloc -N 1 -n 4 -t 10 -p mit_normal
cd /orcd/data/orcd/022/benchmarks/mpi-laplace/work
module load gcc/12.2.0 openmpi/4.1.4
../src/bin-r8/laplace_serial            < ./inp
mpirun -np 4 ../src/bin-r8/laplace_mpi  < ./inp
```

## Analysis

Each run ends with:

```
Max error at iteration 3372 was 0.009995
Total time was 4.088348 seconds.
```

The output file has two of these blocks: serial first, then MPI.

```bash
cd work
for f in */output/*.out; do
  echo "$f $(grep 'Total time' $f | awk '{print $4}' | paste -sd' ')"
done                                   # <file> <serial s> <mpi s>
```

Check that both runs stop at the same iteration (3372). Then compare:

- serial time across nodes: single-core speed;
- MPI time across nodes: 4-core compute plus intra-node message passing;
- speed-up = serial / MPI: about 4 is ideal, typically 3–4.

A node with a much longer time than others of the same CPU type needs a closer
look.
