# mpi-p2p

## Introduction

Inter-node MPI point-to-point benchmark built on the
[OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/).
It runs `osu_bw` (bandwidth) and `osu_latency` between two nodes to
measure the interconnect (InfiniBand) bandwidth in MB/s and latency in
microseconds.

## Installation

OSU micro-benchmarks must be compiled from source per MPI version.
`src/build.sh` does this: it unpacks the OSU tarball, loads the requested
module, then `configure`/`make`/`make install` into
`install/<os>-<version>/` (or `install/<os>-intel-<version>/` for Intel).

```bash
# build.sh <os: c7|r8> <module: openmpi|intel-hpc> <version>
cd src
./build.sh r8 openmpi 4.1.4          # uses mpicc / mpicxx
./build.sh r8 intel-hpc 2025.2.1.44  # uses mpiicx / mpiicpx
```

Submit it as a job with `src/job-build.sh` (a 1-task sbatch wrapper).
Variant scripts (`build-nvhpc.sh`, `build-stage.sh`, …) build CUDA-aware
or staged trees. Source archives come from
<https://mvapich.cse.ohio-state.edu/benchmarks/>; for CUDA-aware builds
see `notes` / `notes.claude` (UCX flags).

At run time, `run/env.sh` loads the matching module stack and puts the
installed OSU binaries on `PATH`:

```bash
# env.sh <os> <openmpi-version>, e.g.
source run/env.sh r8 4.1.4
```

## Usage

### Automated, many runs — `run/`

`run.sh` schedules a 2-node job for **every pair** of the supplied nodes
and runs `osu_bw` + `osu_latency` on each pair.

```bash
cd run
# run.sh "<nodes>" <partition> <reservation|none> <qos>
./run.sh "3506 3507 3508" mit_normal_gpu none unlimited
```

Output lands in `work/<partition>/output/`.

### Single run — `work/`

`work/pt2pt-all-example.sh` shows a single hand-launched pair run; the
many per-site subdirectories under `work/` hold previous campaigns. To
run one pair manually, `source run/env.sh` then `mpirun -n 2 osu_bw`
inside a 2-node allocation.

## Analysis

```bash
cd run
# get-results.sh <partition> <N>
./get-results.sh mit_normal_gpu 2
```

Prints the bandwidth (MB/s) and average latency (µs) at the 4 MiB
(4194304-byte) message size for the most recent N runs.

### Quick health test for one pair — `run/p2p-pair.sh`

Used by check-cluster's `mpi` domain (every Rocky 8 node paired with its
neighbour); also runnable by hand. One `srun --mpi=pmi2` step running a single
`osu_latency` sweep (1 B–4 MiB) with the `r8-4.1.4` build; prints
`P2P_RESULT lat_us=… big_lat_us=… bw_mbs=… exec_s=…` (8 B latency, 4 MiB
one-way bandwidth, benchmark runtime without queue wait).

```bash
./p2p-pair.sh node1363 node1364 [partition=ou_orcd_everything] [build=r8-4.1.4]
```

Notes: `salloc` can't get its allocation back to the login node, and
`sched_system_all` doesn't hand out allocations right now, hence srun +
`ou_orcd_everything`. `install/r8-5.0.8` is linked against nvhpc's
openmpi-3.1.5 — don't use it.

### Which MPI launchers work — `run/launcher-test.sh`

```bash
./launcher-test.sh <mpirun|pmix|pmi2> node1363 node1364 [partition] [build]
```

Runs a 2-rank `osu_latency` (1–8 B) with `mpirun` (inside an `sbatch --wait`
job), `srun --mpi=pmix`, or `srun --mpi=pmi2`; last line is
`LAUNCH_RESULT method=… works=yes|no|untested lat_us=…` (`untested` = Slurm
never started the job). As of 2026-09-23: mpirun works, pmi2 works, pmix does
not (Slurm has no pmix plugin; `srun --mpi=list` = none, cray_shasta, pmi2).

### Launcher matrix — `run/launcher-matrix.sh`

Runs inside a 2-node job and tries every given OpenMPI stack with `mpirun`,
`srun --mpi=pmix` and `srun --mpi=pmi2` (2-rank `osu_latency`). check-cluster's
`mpi` domain submits one job per partition (mit_normal, mit_normal_gpu first,
then all others) and tables the results in `mpi.log` / `summary.md`.

```bash
sbatch -p mit_normal -N 2 --ntasks-per-node=1 run/launcher-matrix.sh \
  "spack-4.1.4|/orcd/software/core/001/spack/modulefiles/gcc/12.2.0|openmpi/4.1.4|r8-4.1.4|/orcd/software/core/001/spack/pkg/openmpi/4.1.4/zuyo6jx"
```

OSU builds for the matrix (built 2026-09-23 with `src/build-ompi-stack.sh`,
which never overwrites an existing build; job: `src/job-build-ompi-stacks.sh`):

| Module | Module path | OSU build |
|---|---|---|
| openmpi/4.1.4 | `/orcd/software/core/001/spack/modulefiles/gcc/12.2.0` | `r8-4.1.4` (existing) |
| openmpi/5.0.8 | `/orcd/software/core/001/spack/modulefiles/gcc/12.2.0` | `r8-spack-ompi-5.0.8` |
| openmpi/5.0.8 | `/orcd/software/core/001/modulefiles` | `r8-core-ompi-5.0.8` |

The core openmpi/5.0.8 links `libhcoll`/`libocoms` from `/opt/mellanox/hcoll`,
which only some nodes have (node1619 yes, node1363 no) — build and run it on
nodes that have it.
