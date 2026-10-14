# cuda

## Introduction

A collection of standalone CUDA programs used to exercise and benchmark
NVIDIA GPUs — matrix multiply (custom kernels and cuBLAS, single/double
precision), SAXPY, dot product, vector add, a π estimator, plus larger
worked examples (CNN, wave/Laplace solvers, multi-GPU). They report
kernel time / throughput and serve as correctness and performance checks
across GPU types (L40S, H100, H200).

Layout:
- `src/` — the main set of kernels (`MatMult*.cu`, `matmul_cuBLAS*.cu`,
  `SAXPY.cu`, `dot.cu`, `CalculatePi.cu`, …) plus the job launcher.
- `cuda-examples/`, `cuSolver/`, `gpu-harvard-workshop/`,
  `github-cuda-ex/` — additional example sets (some with their own README).

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/cuda/src
# build the binaries first (install.md); the old prebuilt ones are corrupt
srun -t 30 -n 8 --gres=gpu:h100:1 -p mit_normal_gpu --mem=10GB \
     bash -c 'module load cuda/12.9.1; ./MatMult_d_h100; ./matmul_cuBLAS_d_h100'
# GPU time: "Cuda code took X seconds" (naive) / "cuBLAS took X milliseconds" (cuBLAS); lower is better
```

## Installation

See [install.md](install.md): build the `src/*.cu` programs with `nvcc`. The old
prebuilt binaries in `src/` are corrupt (zero-filled), so rebuild before running.

## Usage

| Program | What it does |
|---|---|
| `MatMult`, `MatMult_16x16`, `MatMult_16x32` | naive single-precision matmul, 10000², different thread-block shapes |
| `MatMult_d_<gpu>[_16x16\|_16x32]` | the same in double precision |
| `matmul_cuBLAS`, `matmul_cuBLAS_d_<gpu>` | the same product with cuBLAS `Sgemm` / `Dgemm` |
| `SAXPY`, `dot`, `CalculatePi`, `HelloWorld` | small examples |

Each program also runs the same computation serially on the CPU, so it reports
both GPU and CPU times.

### Automated, many runs — `src/run_job.sh`

`run_job.sh` submits one job per node in `nodelist` (all on one `partition`).
Each job runs the single-precision list `exe32p`, then the double-precision list
`exe64p`. Edit those four variables at the top, create the output dirs (see
`install.md`), then:

```bash
cd src
./run_job.sh
```

The output goes to `cuda/out_files/{single,double}_precision/<partition>/<exe>_<node>_.out`.
`SLURM_JOB_ID` is expanded at submit time, when it is empty, so a rerun on the
same node overwrites the file.

### Single run — `src/`

```bash
srun -t 30 -n 8 --gres=gpu:1 -p mit_normal_gpu --mem=10GB --pty bash
cd /orcd/data/orcd/022/benchmarks/cuda/src
./MatMult_d_h100        # or ./matmul_cuBLAS_d_h100, ./SAXPY, ...
```

## Analysis

The naive kernels print `Cuda code took <s> seconds`, the cuBLAS programs
print `cuBLAS took <ms> milliseconds`, and both print the serial CPU time and
`The calculation was correct`. To collect the GPU times:

```bash
cd /orcd/data/orcd/022/benchmarks/cuda/out_files
grep -HE 'Cuda code took|cuBLAS took' */*/*.out | awk '{print $1, $(NF-1), $NF}' | column -t
```

For an N×N matmul, GFLOP/s = 2·N³ / time / 1e9 (N = 10000 → 2000 / time).
Compare the same program across GPU types, and the naive kernels against
cuBLAS, to see how close each gets to peak. Discard any run that prints
`There was a calculation error`.
