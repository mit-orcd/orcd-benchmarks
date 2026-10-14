# mpi-calc-pi

## Introduction

An MPI benchmark that estimates π by numerical integration. It measures
CPU strong-scaling: the same problem is run with an increasing number of
MPI ranks and the wall-clock time is reported, so you can see how a node
scales from 1 rank up to twice its core count.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/mpi-calc-pi/run
./run.sh "3511" mit_normal none unlimited 96     # node3511, 96 cores: ranks 1..192
# when done:
./get-results.sh mit_normal 1                    # rank count + "time taken" for each step
```

## Installation

See [install.md](install.md): one `mpicc` build of `src/calc_pi_mpi.c`.

## Usage

### Automated, many runs — `run/`

Sweeps a list of nodes, submitting one Slurm job per node that runs the
benchmark over a rank list (1, 2, 4, … up to 2× the core count).

```bash
cd run
# run.sh "<nodes>" <partition> <reservation|none> <qos> <cpu_count>
./run.sh "3511 3512" mit_normal none unlimited 96
```

The rank list is 1, 2, 4, … (doubling while 4 × ranks < cores), then cores/2,
cores, 1.5 × cores and 2 × cores, with `mpirun --oversubscribe`. Each job takes
the whole node (`--exclusive`, 30 min limit). The output goes to
`work/<partition>/output/out_full.<node>-<jobid>`. `submit.sh` only calls
`run.sh`.

### Single run — `work/`

`work/job.sh` is a self-contained sbatch script. It runs `calc_pi_mpi_big` at a
fixed rank list (`for i in 48 72 96`) on the old Rocky-8 Open MPI. Edit the
`#SBATCH` lines and the list, then:

```bash
cd work
sbatch job.sh
```

`work/job_full_all_example.sh` and `job_half_all_example.sh` are older copies of
`run/run.sh` with the variables set inside the script (full = up to 2 × cores,
half = up to cores).

## Analysis

Each step of a run prints:

```
Ran with MPI_NUM_THREADS=4
time taken: 59.94688355
pi is approximately 3.1415926535773928, Error is 0.0000000000124003
```

```bash
cd run
./get-results.sh <partition> <N>     # rank count and time for the newest N runs
```

The time measures pure compute with one reduction at the end, so ideal scaling
is time(1)/ranks. Speed-up should stay close to linear up to the physical core
count. With more ranks than cores (hyper-threads or oversubscription) the time
flattens or rises. A node much slower than its peers at the same rank count, or
with no gain from cores/2 to cores, needs a closer look (CPU frequency, BIOS,
power settings). The `Error` value is only a sanity check.
