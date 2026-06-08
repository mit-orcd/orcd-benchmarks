# gpu-burn-r8

## Introduction

A multi-GPU CUDA stress test based on
[gpu-burn](https://github.com/wilicc/gpu-burn) (http://wili.cc/blog/gpu-burn.html).
It runs heavy matrix-multiply kernels on every GPU for a fixed duration
and reports per-GPU Gflop/s plus any compute errors, so it is used both
as a burn-in/health check and a rough throughput check. Runs are done in
three modes: tensor-core FP (`-tc`), standard single precision, and
doubles (`-d`).

## Installation

The source is in this repo (`gpu_burn-drv.cpp`, `compare.cu`, `Makefile`).
Build with CUDA:

```bash
module load cuda
make                 # produces ./gpu_burn   (override arch: make COMPUTE=8.0)
```

Upstream / Docker build instructions: <https://github.com/wilicc/gpu-burn>.

## Usage

The binary is `./gpu_burn [options] [seconds]` (`-tc` tensor cores,
`-d` doubles, `-m` memory, `-i N` only GPU N, `-l` list GPUs).

### Automated, many runs — `run/`

`run.sh` submits one job per node; each runs the three modes for 300 s
and writes a file per mode.

```bash
cd run
# args: "<nodes>" <partition> <reservation|none> <qos> <cpu_count> <gpu_type> <gpu_count>
./run.sh "3100 3101" mit_normal_gpu none unlimited 8 l40s 4
```

Output lands in `<partition>/output-<gpu_type>/`.

### Single run — root dir

Get an interactive GPU session, then run directly, or use the standalone
scripts:

```bash
srun -t 60 -n 8 --gres=gpu:4 -p mit_normal_gpu --mem=10GB --pty bash
./gpu_burn -tc 300        # tensor cores, 300 s

# or a packaged 3-mode run on the current host:
./run-standard-gpu-burn-tests.sh
```

## Analysis

```bash
cd run
# get-results.sh <partition> <N> <gpu_type>
./get-results.sh mit_normal_gpu 2 l40s
```

Prints the per-GPU summary (`100.0%` proc'd line → Gflop/s and error
count) from the most recent runs. A healthy GPU finishes with `OK` and
zero errors; `FAULTY`/`Killing` indicates a failing GPU.
