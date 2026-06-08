# openmp

## Introduction

An OpenMP benchmark that estimates π with a parallel reduction. It
measures CPU thread-scaling on a single node: the same loop is run with
an increasing number of threads (1, 2, 4, … up to 2× the core count) and
the wall-clock time per thread count is reported.

## Installation

The source is in `src/` (`pi_omp.c`); nothing needs to be downloaded.

```bash
module load gcc/12.2.0-x86_64
cd src
gcc -O3 -fopenmp pi_omp.c -o pi_omp
```

## Usage

### Automated, many runs — `run/`

Sweeps a list of nodes, submitting one Slurm job per node that runs
`pi_omp` over a thread list derived from the core count.

```bash
cd run
# run.sh "<nodes>" <partition> <reservation|none> <qos> <cpu_count>
./run.sh "3511 3512" mit_normal none unlimited 96
```

`run-test.sh` is a quick smoke-test variant. Output lands in
`work/<partition>/output/`.

### Single run — `work/`

`work/job.sh` is a self-contained sbatch script for one node. Edit the
`#SBATCH` partition/cores and the thread list in the loop, then:

```bash
cd work
sbatch job.sh        # loops OMP_NUM_THREADS over the listed values
```

## Analysis

```bash
cd run
# get-results.sh <partition> <N>
./get-results.sh mit_normal 4
```

Prints the thread count (`THREADS`) and elapsed `time` from each run;
compare times across thread counts to judge scaling.
