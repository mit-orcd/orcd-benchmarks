# all-bench

## Introduction

A driver that runs the whole benchmark suite across a set of nodes in one
shot. It does not contain a benchmark of its own — instead it loops over a
chosen list of benchmarks (e.g. `openmp`, `mpi-calc-pi`, `mpi-p2p`,
`gpu-burn-r8`, `nvidia-hpc-benchmarks`, `nccl-tests`) and calls each
benchmark's own `run/run.sh` (and `run-2node.sh` where present), then
collects everything with each benchmark's `get-results.sh`. It is the
top-level entry point for node acceptance / partition sweeps.

## Installation

Nothing to build here. Each target benchmark must already be built in its
own directory (see the per-benchmark READMEs). The driver only needs the
sibling benchmark dirs present under `/orcd/data/orcd/022/benchmarks`.

## Usage

All scripts are in the root dir. There is no separate single-run mode —
the `run-*.sh` scripts *are* the automated multi-node, multi-benchmark
drivers; for a single benchmark on one node use that benchmark's own
`work/`/`run/` scripts.

### Automated, many runs — root `run-*.sh`

Each `run-*.sh` is a preset for a partition/hardware target. Edit the
header variables (`nodes`, `partition`, `reservation`, `qos`, `cpus`,
`gpu_type`, `gpus`) and the `all_bench` list, then run it:

```bash
./run-all.sh                # generic GPU-node preset
./run-normal.sh             # CPU-node (mit_normal) preset
./run-normal-gpu-h200.sh    # H200 GPU-node preset (and -h100 / -l40s variants)
# many site-specific presets also exist: run-pi_*.sh, run_ou_*.sh, run-bcs-*.sh, ...
```

Each calls `<bench>/run/run.sh "<nodes>" <partition> <reservation> <qos> <cpus> <gpu_type> <gpus>`.

## Analysis

The `get-results-*.sh` scripts mirror the `run-*.sh` presets. Set the
`partition`, `lines` (how many recent jobs), `gpu_type`, and `all_bench`
list at the top, then run:

```bash
./get-results-all.sh            # matches run-all.sh
./get-results-normal.sh         # CPU presets
./get-results-pi_mbathe.sh      # site-specific presets, etc.
```

Each dispatches to `<bench>/run/get-results.sh`, so the reported metric is
whatever that benchmark prints (scaling time, bandwidth, GFLOP/s, …).
