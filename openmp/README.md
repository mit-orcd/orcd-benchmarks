# openmp

## Introduction

An OpenMP benchmark that estimates π with a parallel reduction. It
measures CPU thread-scaling on a single node: the same loop is run with
an increasing number of threads (1, 2, 4, … up to 2× the core count) and
the wall-clock time per thread count is reported.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/openmp/run
./run.sh "3511" mit_normal none unlimited 96     # node3511, 96 cores: 1..192 threads
# when done:
./get-results.sh mit_normal 1                    # thread count + "time to compute" for each step
```

## Installation

See [install.md](install.md): one `gcc -fopenmp` build of `src/pi_omp.c`.

## Usage

### Automated, many runs — `run/`

Sweeps a list of nodes, submitting one Slurm job per node that runs
`pi_omp` over a thread list derived from the core count.

```bash
cd run
# run.sh "<nodes>" <partition> <reservation|none> <qos> <cpu_count>
./run.sh "3511 3512" mit_normal none unlimited 96
```

The thread list is 1, 2, 4, … (doubling while 4 × threads < cores), then
cores/2, cores, 1.5 × cores and 2 × cores. Each job takes the whole node
(`--exclusive`, 30 min). The output goes to
`work/<partition>/output/out_full.<node>-<jobid>`. `run-test.sh` is a quick
smoke-test variant.

### Single run — `work/`

`work/job.sh` is a self-contained sbatch script for one node. Edit the
`#SBATCH` partition/cores and the thread list in the loop, then:

```bash
cd work
sbatch job.sh        # loops OMP_NUM_THREADS over the listed values
```

## Analysis

Each step prints:

```
Ran with OMP_NUM_THREADS=2
pi = 3.141592653588666
time to compute = 11.9297 seconds
```

`real`/`user`/`sys` lines from `time` follow it.

```bash
cd run
./get-results.sh <partition> <N>      # thread count and times for the newest N runs
```

The loop is pure compute with no memory traffic, so ideal scaling is
time(1)/threads. Expect close to linear speed-up up to the physical core count,
then little or no gain with hyper-threads. Compare nodes of the same CPU type
at the same thread count. A node that is clearly slower, or stops scaling
early, may have reduced clocks, disabled cores or a BIOS/power-profile
difference.
