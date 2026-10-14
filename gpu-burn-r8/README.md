# gpu-burn-r8

## Introduction

A multi-GPU CUDA stress test based on
[gpu-burn](https://github.com/wilicc/gpu-burn) (http://wili.cc/blog/gpu-burn.html).
It runs heavy matrix-multiply kernels on every GPU for a fixed duration
and reports per-GPU Gflop/s plus any compute errors, so it is used both
as a burn-in/health check and a rough throughput check. Runs are done in
three modes: tensor-core FP (`-tc`), standard single precision, and
doubles (`-d`).

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/gpu-burn-r8/run
./run.sh "3100" mit_normal_gpu none unlimited 8 l40s 4     # 1 node: tc32, std32, d64, 300 s each
# when done (~16 min):
./get-results.sh mit_normal_gpu 1 l40s
grep -A8 'Tested' ../mit_normal_gpu/output-l40s/*_node3100-<jobid>.out     # every GPU must say OK
```

## Installation

See [install.md](install.md): `make` in this dir (with `CUDAPATH` overridden)
produces `./gpu_burn`.

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

Each output file is `<partition>/output-<gpu_type>/<mode>_<node>-<jobid>.out`,
where `<mode>` is `tc32` (tensor cores), `std32` (FP32) or `d64` (FP64). Each
progress line holds, per GPU, the iterations done, Gflop/s, error count and
temperature:

```
100.0%  proc'd: 27768 (110036 Gflop/s) - 27924 (110572 Gflop/s) - ...   errors: 0 - 0 - ...   temps: 63 C - 62 C - ...
```

```bash
cd run
./get-results.sh <partition> <N> <gpu_type>    # final (100.0%) line of the newest N runs x 3 modes
```

The file ends with a verdict for each GPU:

```bash
grep -A9 Tested ../<partition>/output-<gpu_type>/*.out | grep -E 'GPU [0-9]+:' | grep -v OK   # empty = all passed
```

A GPU passes when it is `OK` with `errors: 0`. `FAULTY` or a non-zero error count
means the GPU is bad. The `Killing processes ...` lines are the normal shutdown
at the end of every run. Gflop/s should agree
within a few percent across the GPUs of one node and across nodes of the same
type. A low outlier usually means throttling: check `temps`.
