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

## Installation

Build the individual programs with `nvcc` (cuBLAS variants link
`-lcublas`):

```bash
module load cuda/12.4.0
cd src
nvcc -O3 -arch=sm_90 MatMult.cu        -o MatMult          # sm_90 = H100/H200; L40S = sm_89
nvcc -O3 -arch=sm_90 matmul_cuBLAS.cu  -o matmul_cuBLAS -lcublas
```

Prebuilt per-GPU binaries already exist in `src/` (e.g. `MatMult_d_h100`,
`MatMult_d_l40s`).

## Usage

### Automated, many runs — `src/run_job.sh`

Submits one Slurm job per node and runs a configured list of
single-precision (`exe32p`) and double-precision (`exe64p`) executables.
Edit `nodelist`, `partition`, and the executable lists at the top, then:

```bash
cd src
./run_job.sh
```

Output is written per executable to `out_files/<precision>/<partition>/`.

### Single run — `src/`

Get an interactive GPU session and run a binary directly:

```bash
srun -t 30 -n 8 --gres=gpu:1 -p mit_normal_gpu --mem=10GB --pty bash
module load cuda/12.4.0
cd src
./MatMult            # or ./matmul_cuBLAS_d_h100, ./SAXPY, etc.
```

## Analysis

Each program prints its own timing/throughput (e.g. matmul GFLOP/s or
elapsed kernel time) to stdout, captured in the `.out` files under
`src/out_files/`. Compare the same executable across GPU types, and
custom kernels vs. the cuBLAS variant, to gauge achieved vs. peak
performance.
