# orcd-benchmarks

A collection of HPC benchmarks used on the MIT ORCD Slurm cluster to
characterize CPU, GPU, network, and storage performance.

Each subdirectory contains the run scripts and helpers for one benchmark.
Actual benchmark source code, prebuilt binaries, container images, and raw
job output are **not** stored in this repository — rebuild or download them
locally before running.

## Layout

| Directory              | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `py-all-bench/`        | Python automation: submit / analyze across all benchmarks |
| `all-bench/`           | Original bash orchestration scripts                    |
| `openmp/`              | OpenMP pi calculation (CPU)                            |
| `mpi-calc-pi/`         | MPI pi calculation (CPU)                               |
| `mpi-p2p/`             | OSU MPI point-to-point bandwidth / latency             |
| `numpy/`               | NumPy matrix-multiply (CPU)                            |
| `gpu-burn-r8/`         | Multi-GPU CUDA stress test                             |
| `nccl-tests/`          | NVIDIA NCCL collective performance                     |
| `nvidia-hpc-benchmarks/` | NVIDIA HPC-Benchmarks container (HPL, STREAM, …)     |
| `gpu-fryer/`           | GPU stress + memory / precision sweep                  |
| `megatron-lm/`         | Megatron-LM GPT pre-training throughput                |

## Quick start

All benchmarks in `py-all-bench/` can be submitted in one command:

```
module load miniforge/24.3.0-0      # Python 3.7+ required
cd py-all-bench
python bench_submit.py --all-bench \
    --nodes 3506 3507 \
    --partition mit_normal_gpu --qos unlimited \
    --gpu-type l40s --gpus 4
```

Analyze the most recent results the same way:

```
python bench_analyze.py --all-bench \
    --partition mit_normal_gpu --gpu-type l40s --num-results 2
```

See each benchmark's own `README.md` for standalone usage and
`py-all-bench/README.md` for the full automation interface.
