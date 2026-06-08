# mpi-p2p

## Introduction

Inter-node MPI point-to-point benchmark built on the
[OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/).
It runs `osu_bw` (bandwidth) and `osu_latency` between two nodes to
measure the interconnect (InfiniBand) bandwidth in MB/s and latency in
microseconds.

## Installation

OSU micro-benchmarks are built per MPI version under `install/<tag>/`.
`run/env.sh` loads the matching module stack and puts the OSU binaries on
`PATH`:

```bash
# env.sh <os> <openmpi-version>, e.g.
source run/env.sh r8 4.1.4
```

Source archives come from <https://mvapich.cse.ohio-state.edu/benchmarks/>.
For CUDA-aware builds see `notes` / `notes.claude` (UCX flags).

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
