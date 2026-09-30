# orcd-benchmarks

A collection of HPC benchmarks used on the MIT ORCD Slurm cluster to
characterize CPU, GPU, network, and storage performance.

Each subdirectory contains the run scripts and helpers for one benchmark.
Actual benchmark source code, prebuilt binaries, container images, and raw
job output are **not** stored in this repository — rebuild or download them
locally before running. The B200 / DGX directories also keep their result
summaries (`*.md`) and the outputs of successful runs.

## Layout

### Drivers

| Directory              | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `all-bench/`           | Bash driver: `run-all.sh [bench ...]` runs any set of benchmarks below on a set of nodes |
| `py-all-bench/`        | Python automation: submit / analyze across all benchmarks |

### CPU, memory, MPI, storage

| Directory              | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `openmp/`              | OpenMP pi calculation (CPU)                            |
| `mpi-calc-pi/`         | MPI pi calculation (CPU)                               |
| `mpi-p2p/`             | OSU MPI point-to-point bandwidth / latency             |
| `mpi-laplace/`         | MPI 2-D Laplace solver (halo exchange)                 |
| `mpi-io/`              | MPI-IO parallel file read / write                      |
| `numpy/`               | NumPy matrix-multiply (CPU)                            |
| `stream-amd/`          | STREAM memory bandwidth, AMD EPYC build                |
| `stream-intel/`        | STREAM memory bandwidth, Intel build                   |
| `fio/`                 | Filesystem bandwidth / IOPS                            |
| `dataloader/`          | PyTorch DataLoader I/O throughput                      |

### GPU

| Directory              | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `cuda/`                | Standalone CUDA programs (matmul, cuBLAS, …)           |
| `gpu-burn-r8/`         | Multi-GPU CUDA stress test                             |
| `gpu-fryer/`           | GPU stress + precision sweep (fp32 / bf16 / fp8)       |
| `nccl-tests/`          | NVIDIA NCCL collective performance                     |
| `nvidia-hpc-benchmarks/` | NVIDIA HPC-Benchmarks container (HPL, HPCG, STREAM)  |
| `megatron-lm/`         | Megatron-LM GPT pre-training throughput (L40S, H200)   |

### Node-specific campaigns

| Directory              | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `b200-nodes/`          | B200 (Rocky 8): gpu-fryer, NCCL, SHARP, ib_write_bw, Megatron-LM ~7B |
| `b200-ubuntu/`         | B200 on Ubuntu 24.04 (node5700/5701): gpu-fryer, NCCL, Ubuntu vs Rocky 8 |
| `b200-kimi/`           | Kimi-K3 vLLM serving on 2 × 8 B200, compared with MI355X |
| `dgx-nodes/`           | NCCL on the Ubuntu DGX H100 nodes                      |

`chris-*` directories hold separate notes and tests (fio, GPU functional
tests, NCCL, OpenMP on GPU).

## Quick start

Bash driver — set the node and GPU variables at the top of
`all-bench/run-all.sh`, then:

```
cd all-bench
./run-all.sh                         # the default benchmark list in the script
./run-all.sh gpu-burn-r8 nccl-tests  # only these
./run-all.sh b200-nodes              # B200 set
```

Or load the bash driver as a module (group `sched_mit_orcd` only) and set the
run through `BENCH_*` environment variables instead of editing the script:

```
module use /orcd/compute/orcd/011/team-software/modulefiles
module load orcd-benchmark           # run-all.sh, get-results-*.sh on PATH
BENCH_NODES="3511 3512" BENCH_PARTITION=mit_normal_gpu \
    BENCH_GPU_TYPE=l40s BENCH_GPUS=4 run-all.sh gpu-burn-r8 nccl-tests
get-results-all.sh                   # collect results
```

Variables: `BENCH_NODES BENCH_PARTITION BENCH_RESERVATION BENCH_QOS BENCH_CPUS
BENCH_GPU_TYPE BENCH_GPUS`; `module help orcd-benchmark` for details.

Python automation — all benchmarks in `py-all-bench/` in one command:

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

See each benchmark's own `README.md` for standalone usage,
`all-bench/README.md` for the bash driver, and `py-all-bench/README.md` for the
Python automation.
