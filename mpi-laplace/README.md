# mpi-laplace

## Introduction

An MPI benchmark that solves the 2-D Laplace equation by Jacobi
iteration on a grid. It exercises nearest-neighbour halo exchange and
reduction, so it measures combined compute + communication scaling
across MPI ranks. A serial version is included for a baseline. (A small
calc-pi MPI/serial pair also lives in `src/` as a sanity check.)

## Installation

The source is in `src/`; nothing needs to be downloaded.

```bash
module load gcc/12.2.0-x86_64 openmpi/4.1.4-pmi-ucx-x86_64
gcc   -O3 -lm ../src/laplace_serial.c -o laplace_serial
mpicc -O3 -lm ../src/laplace_mpi.c    -o laplace_mpi
```

Prebuilt binaries live under `src/bin-r8/` (matched to OpenMPI 4.1.4).

## Usage

The solver reads its grid/iteration parameters from an input file
(`work/inp`) on stdin.

### Automated, many runs — `work/job_all_example.sh`

Submits one Slurm job per node; each runs the serial baseline then the
MPI solver. Edit the `nodes` and `partition` variables at the top, then:

```bash
cd work
./job_all_example.sh
```

Output lands in `work/<partition>/output/`.

### Single run — `work/`

Inside a node allocation, run a fixed rank count against the input:

```bash
cd work
module load gcc/12.2.0-x86_64 openmpi/4.1.4-pmi-ucx-x86_64
../src/bin-r8/laplace_serial          < ./inp
mpirun -np 4 ../src/bin-r8/laplace_mpi < ./inp
```

## Analysis

The solver prints the iteration count and the elapsed solve time at the
end of each run. Compare the MPI time across rank counts (and against the
serial baseline) to judge scaling; results are kept per partition under
`work/<partition>/`.
