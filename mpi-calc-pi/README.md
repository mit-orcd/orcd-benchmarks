# mpi-calc-pi

## Introduction

An MPI benchmark that estimates π by numerical integration. It measures
CPU strong-scaling: the same problem is run with an increasing number of
MPI ranks and the wall-clock time is reported, so you can see how a node
scales from 1 rank up to twice its core count.

## Installation

The source is in `src/`; nothing needs to be downloaded. Build with an
MPI compiler:

```bash
module load gcc/12.2.0 openmpi/4.1.4
cd src
mpicc -O3 calc_pi_mpi.c     -o calc_pi_mpi
mpicc -O3 calc_pi_mpi_big.c -o calc_pi_mpi_big   # larger work size
```

## Usage

### Automated, many runs — `run/`

Sweeps a list of nodes, submitting one Slurm job per node that runs the
benchmark over a rank list (1, 2, 4, … up to 2× the core count).

```bash
cd run
# run.sh "<nodes>" <partition> <reservation|none> <qos> <cpu_count>
./run.sh "3511 3512" mit_normal none unlimited 96
```

`submit.sh` is a thin wrapper that calls `run.sh` with preset arguments.
Output lands in `work/<partition>/output/`.

### Single run — `work/`

`work/job.sh` is a self-contained sbatch script for one node. Edit the
`#SBATCH` partition/cores and the rank list in the loop, then:

```bash
cd work
sbatch job.sh
```

## Analysis

```bash
cd run
# get-results.sh <partition> <N>   -> most recent N output files
./get-results.sh mit_normal 4
```

It prints the rank count (`THREADS`) and elapsed `time` from each run;
compare times across rank counts to judge scaling.
