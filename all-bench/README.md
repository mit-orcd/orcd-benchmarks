# all-bench

## Introduction

A driver that runs a set of benchmarks across a set of nodes in one shot. It has
no benchmark of its own: `run-all.sh` loops over the chosen benchmarks and submits
each one's Slurm jobs, then `get-results-*.sh` collects the numbers. It is the
top-level entry point for node acceptance and partition sweeps.

Benchmarks `run-all.sh` can run:

| Benchmark | Node type | What it measures |
|---|---|---|
| `openmp` | CPU / GPU node | OpenMP pi, thread-count sweep |
| `mpi-calc-pi` | CPU / GPU node | MPI pi, rank-count sweep |
| `mpi-p2p` | node pairs | OSU `osu_bw` + `osu_latency` |
| `gpu-burn-r8` | GPU | gpu_burn TF32 / FP32 / FP64, 300 s each |
| `nvidia-hpc-benchmarks` | GPU | HPL, HPL-MxP, HPCG, STREAM (NVIDIA container) |
| `nccl-tests` | GPU, 1 node + pairs | NCCL `sendrecv_perf` bus bandwidth |
| `megatron-lm` | L40S / H200, 1 node + pairs | Megatron-LM GPT pretrain TFLOP/s per GPU |
| `b200-nodes` | B200, 1 node + pairs | gpu-fryer, NCCL 1-node/2-node, Megatron-LM 1-node/2-node |
| `b200-kimi` | B200, 1–2 nodes | Kimi-K3 vLLM serving, concurrency 1–64 |

### b200-nodes vs the general benchmark dirs

`b200-nodes` re-runs three of the general benchmarks on B200. Two are the same
benchmark; Megatron-LM is not.

| b200-nodes | General dir | Same? | Details |
|---|---|---|---|
| gpu-fryer | `gpu-fryer` | **Same** | Same image (`gpu-fryer_1.1.0.sif`), same fp32 / bf16 / fp8 runs, 300 s each. b200-nodes stresses all 8 GPUs of an exclusive node (`gpu-fryer/job.sh` asks for 1 GPU) and falls back to `--unsquash` where FUSE is blocked. Per-GPU TFLOP/s are comparable. |
| NCCL 1-node / 2-node | `nccl-tests` | **Same test, different build** | Same nccl-tests binaries and sizes (1M–16G, factor 4, default `sendrecv`). B200 needs CUDA 13, so b200-nodes uses `build-nvhpc-26.1` with the nvhpc/26.1 HPC-X OpenMPI; `nccl-tests/run` uses `build-nvhpc-24.5-ompi-5.0.8`. The b200 2-node job pins `NCCL_IB_HCA` to the 8 NDR rails, bootstraps MPI over TCP, and takes GPUs/node as an argument (`nccl-tests` 2-node is fixed at 1). Bus bandwidth is comparable. |
| Megatron-LM 1-node / 2-node | `megatron-lm` | **Different** | Same container (`pytorch_26.02-py3.sif`), same Megatron-LM tree, mock data, micro-batch 4, global batch 128 × GPUs, bf16, TP = PP = 1, 100 iterations. The model size differs by GPU memory: L40S 24 layers / hidden 2048 / FFN 8192; H200 24 layers / hidden 4096 / FFN 16384; B200 36 layers / hidden 4096 / FFN 14336 (~7B, matching the AICR B200 reference). TFLOP/s per GPU is not comparable across GPU types. |

The other b200-nodes tests (`ib_write_bw` in `job-ibwrite-*.sh`, NCCL with SHARP
in `job-nccl-2node-sharp*.sh`, the `-max` and 3-node Megatron-LM variants) have no
general counterpart and are not run by `run-all.sh`.

`b200-kimi` has no counterpart among the general dirs; it compares against the
MI355X run in `amd-benchmarks/amd-cloud`.

## Installation

Nothing to build here. Each target benchmark must already be built or pulled in
its own directory (see its README). `run-all.sh` only needs the benchmark dirs
under `/orcd/data/orcd/022/benchmarks`.

## Usage

All scripts are in this dir.

### Automated, many runs — `run-all.sh`

Set the node and resource variables at the top of `run-all.sh`, then pick the
benchmarks on the command line or with the `all_bench` default:

```bash
./run-all.sh                          # run the all_bench list in the script
./run-all.sh gpu-burn-r8 nccl-tests   # run only these
./run-all.sh b200-nodes               # B200 set, e.g. with nodes="5500 5502" gpus=8
```

| Variable | Example | Meaning |
|---|---|---|
| `nodes` | `"3511 3512"` | node numbers (`nodeNNNN`), space separated |
| `partition` | `mit_normal_gpu` | Slurm partition, also the output dir name |
| `reservation` | `none` | Slurm reservation; `none` = no reservation |
| `qos` | `unlimited` | Slurm QOS |
| `cpus` | `48` | cores per node; sets the openmp / mpi-calc-pi sweep |
| `gpu_type` | `l40s` | GPU type (`l40s`, `h100`, `h200`, `b200`, …) |
| `gpus` | `4` | GPUs per node |
| `all_bench` | `"nccl-tests"` | default benchmark list, overridden by command-line args |

How each benchmark is launched:

- `openmp`, `mpi-calc-pi`, `mpi-p2p`, `gpu-burn-r8`, `nvidia-hpc-benchmarks`,
  `nccl-tests`: `<bench>/run/run.sh "<nodes>" <partition> <reservation> <qos> <cpus> <gpu_type> <gpus>`,
  plus `run-2node.sh` with the same arguments if present (nccl-tests).
- `megatron-lm`: `Megatron-LM/job.sh` on each node, then on each node pair,
  with global batch = 128 × total GPUs. Only `gpu_type` `l40s` or `h200` (the
  model is chosen by GPU type); other types are skipped.
- `b200-nodes`: `job-gpu-fryer.sh` and `job-nccl-1node.sh` on all nodes,
  `job-megatron-1node.sh` on each node, then `job-nccl-2node.sh` and
  `job-megatron-2node.sh` on each node pair, all with `gpus` GPUs per node. The
  partition is fixed to `mit_testing` in those scripts; `partition`,
  `reservation`, `qos` are not used.
- `b200-kimi`: runs `b200-kimi/chain.sh` (gate → 2-node verify → 1-node attempt
  → 2-node TP8 × PP2 sweep → summary). Nodes, reservation and account come from
  `b200-kimi/common/env.sh`, not from `run-all.sh`.

`run-*.sh` / `run_*.sh` are older presets for specific partitions and groups
(`run-normal.sh` for CPU nodes, `run-normal-gpu-h200.sh`, `run-pi_*.sh`, …). They
take the same variables and call the same `run/run.sh` scripts.

### Single run

Use a benchmark's own scripts (`<bench>/run/`, `work/`, or its root dir) for one
run on one node; see that benchmark's README.

## Analysis

`get-results-*.sh` mirror the `run-*.sh` presets. Set `partition`, `lines` (how
many recent jobs), `gpu_type` and `all_bench` at the top, then run:

```bash
./get-results-all.sh            # matches run-all.sh
./get-results-normal.sh         # CPU presets
```

Each calls `<bench>/run/get-results.sh`, so the metric is whatever that
benchmark reports (time, bandwidth, GFLOP/s, …). The benchmarks without a
`run/` dir have their own analyzers:

- `megatron-lm`: TFLOP/s per GPU on the `--log-throughput` lines in
  `megatron-lm/Megatron-LM/output/`.
- `b200-nodes`: `analyze-gpu-fryer.py`, `analyze-nccl-1node.py`,
  `analyze-nccl-2node.py`, `analyze-megatron.py`, which write `summary.md`
  into `out-gpu-fryer/`, `out-nccl-1node/`, `out-nccl-2node/`, `output-megatron/`.
- `b200-kimi`: written automatically by the chain to
  `results/kimi-k3-base-b200.md` and `results/RUN-SUMMARY.md`.
