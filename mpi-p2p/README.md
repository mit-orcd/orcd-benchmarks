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
