# nvidia-hpc-benchmarks

## Introduction

NVIDIA's [HPC-Benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks)
container provides GPU builds of HPL, HPL-MxP, HPCG and STREAM. The scripts here
run **HPL** (FP64 dense LU / Linpack) on 1, 2, 4 and 8 GPUs of one node. It
reports GFLOP/s in total and per GPU, which shows FP64 throughput and
multi-GPU scaling. The HPL-MxP, HPCG and STREAM commands are in
`run/execute-all.sh` but commented out.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/nvidia-hpc-benchmarks/run
./run.sh "3300" mit_normal_gpu none unlimited 16 h200 8      # HPL on 1, 2, 4, 8 GPUs of node3300
# when done:
./get-results.sh mit_normal_gpu 1 h200                       # total and per-GPU GFLOP/s for each GPU count
```

## Installation

See [install.md](install.md): pull `hpc-benchmarks_25.04.sif` into `run/`.

## Usage

### Automated, many runs — `run/`

`run.sh` submits one exclusive job per node (`--mem=1000GB`; lower it for nodes
with less memory). Each job runs `execute-all.sh <image> <gpu_count>`, which
calls `/workspace/hpl.sh` with the in-image deck `HPL-<n>GPU(s).dat` for
n = 1, 2, 4, 8, up to `gpu_count`. Run it from `run/`, because the job calls
`./execute-all.sh`.

```bash
cd run
# run.sh "<nodes>" <partition> <reservation|none> <qos> <cpu_count> <gpu_type> <gpu_count>
./run.sh "3100 3101" mit_normal_gpu none unlimited 8 l40s 4
```

The output goes to `<partition>/output-<gpu_type>/<node>-<jobid>.out`.
`execute-h200.sh` (H200, larger N) and `execute-mig.sh` (MIG slices) are
variants, submitted by `submit-h200.sh` and `submit-mig.sh`. To also run
HPL-MxP, HPCG or STREAM, uncomment their lines in `execute-all.sh`.

### Single run — `run/job.sh`

`run/job.sh` is a self-contained sbatch script that runs the same 1/2/4/8-GPU
HPL sequence. Edit the partition and GPU type/count in its header, then:

```bash
cd run
sbatch job.sh
```

The root `run.sh`, `job.sh`, `run-2nodes.sh` and the scripts under
`<partition>/` are older per-site versions (24.03 image).

## Analysis

HPL prints one result line per GPU count, then a residual check:

```
T/V                N    NB     P     Q         Time          Gflops (   per GPU)
WC0           136192  1024     2     1        18.13       9.288e+04 ( 4.644e+04)
||Ax-b||_oo/(...)=   0.000263094958 ...... PASSED
```

```bash
cd run
./get-results.sh <partition> <N> <gpu_type>     # total and per-GPU Gflops of the WC0 lines of the newest N runs
grep -E '=+ [0-9]+ GPU|WC0|PASSED|FAILED' ../<partition>/output-<gpu_type>/<file>
```

A run counts only if the residual check says `PASSED`. Per-GPU Gflops should
stay nearly flat from 1 to 8 GPUs (good scaling), and should be a large fraction
of the GPU's FP64 tensor peak (H200 ≈ 47 TFLOP/s per GPU measured here). Consumer
and inference GPUs (L40S, RTX) have little FP64, so their HPL numbers are low by
design.
