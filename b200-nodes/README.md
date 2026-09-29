# b200-nodes

## Introduction

Benchmarks for the NVIDIA B200 nodes running Rocky 8 (node5500–5502 by default;
the rail tests also cover node5602/5702/5800/5802-c1): 8 × B200 per node,
NVLink 5 / NVSwitch inside a node, 8 × NDR 400 Gb/s InfiniBand rails between
nodes. Slurm partition **`mit_testing`**.

| Benchmark | What it measures |
|---|---|
| gpu-fryer | Sustained TFLOP/s per GPU, fp32 / bf16 / fp8, 300 s each |
| NCCL 1-node / 2-node | Collective bus bandwidth over NVLink and over InfiniBand |
| NCCL SHARP | all_reduce with InfiniBand switch offload (SHARP) vs plain ring |
| ib_write_bw | Raw RDMA bandwidth per rail, host vs GPU memory, one rail or all 8 at once |
| Megatron-LM | GPT pre-training TFLOP/s per GPU, 1, 2 or 3 nodes |

Same as, or different from, the general benchmark dirs:

| Here | General dir | Same? |
|---|---|---|
| gpu-fryer | `../gpu-fryer` | **Same**: same image and fp32 / bf16 / fp8 × 300 s; here all 8 GPUs of the node |
| NCCL 1-node / 2-node | `../nccl-tests` | **Same test, different build**: CUDA 13 `build-nvhpc-26.1` (B200 needs it) vs `build-nvhpc-24.5-ompi-5.0.8`; 2-node pins the 8 NDR rails |
| Megatron-LM | `../megatron-lm` | **Different model**: 36 layers / hidden 4096 / FFN 14336 (~7B) here, vs 24/2048/8192 on L40S and 24/4096/16384 on H200, sized to GPU memory. Same container, batch rule and iterations |

The ~7B Megatron-LM model matches the AICR B200 reference, so results compare
directly with it.

## Installation

Nothing to build. The scripts load their own modules and use prebuilt pieces
from the other benchmark dirs:

- NCCL: `../nccl-tests/build-nvhpc-26.1` (module `nvhpc/26.1`, CUDA 13, HPC-X OpenMPI)
- gpu-fryer: `../gpu-fryer/gpu-fryer_1.1.0.sif` (module `apptainer/1.4.2`)
- Megatron-LM: `../megatron-lm/Megatron-LM` and `../megatron-lm/imag/pytorch_26.02-py3.sif`

## Usage

### Automated, many runs — `../all-bench/run-all.sh`

```bash
cd ../all-bench
# set nodes="5500 5502" gpu_type=b200 gpus=8 at the top, then
./run-all.sh b200-nodes
```

This submits gpu-fryer, NCCL 1-node and Megatron-LM 1-node on every node, and
NCCL 2-node and Megatron-LM 2-node on every node pair.

### Single runs — scripts in this dir

The `job-*.sh` scripts submit to Slurm (one job per node, or per GPU count);
the `run-*.sh` scripts run directly after `ssh` to a node.

```bash
./job-gpu-fryer.sh [nodes] [seconds]                    # default node5500 node5502, 300 s
./job-nccl-1node.sh [nodes] [collectives] [ngpus]       # default sendrecv, all GPUs
sbatch job-nccl-2node.sh [collectives] [gpus_per_node]  # pinned to node5500,node5502; override with -w
./job-megatron-1node.sh [node] [ngpus]                  # no ngpus: scan 1..8, one job each
./job-megatron-2node.sh [node,node] [ngpus]             # same, 2 nodes
```

Collectives: `sendrecv allreduce allgather reducescatter reduce broadcast
alltoall gather scatter hypercube`, a comma list, or `all`.

Other tests:

| Script | Purpose |
|---|---|
| `job-megatron-3node.sh` | Megatron-LM ~7B on 3 nodes, same batch rule |
| `job-megatron-1node-max.sh`, `-2node-max.sh` | ~5B model with full recompute for maximum TFLOP/s; `bf16` or `fp8` (not comparable to the reference) |
| `job-nccl-2node-sharp.sh` | all_reduce, ring vs SHARP, back-to-back in one allocation |
| `job-nccl-2node-sharp-aicr.sh` | same, with the AICR cluster's SHARP environment |
| `job-ibwrite-2node.sh`, `-1node.sh` | GPUDirect RDMA `ib_write_bw` on mlx5_4 + GPU0 |
| `job-ibwrite-rails.sh` | each of the 8 rails in turn, host vs GPU memory |
| `job-ibwrite-concurrent.sh` | all 8 rails at once, aggregate bandwidth |

Do not run gpu-fryer and NCCL on the same node at the same time. Slurm output
from the `job-*.sh` wrappers goes to `slurm-logs/`.

## Analysis

| Analyzer | Reads | Writes |
|---|---|---|
| `./analyze-gpu-fryer.py` | `out-gpu-fryer/` | `out-gpu-fryer/summary.md` |
| `./analyze-nccl-1node.py` | `out-nccl-1node/` | `out-nccl-1node/summary.md` |
| `./analyze-nccl-2node.py` | `out-nccl-2node/` | `out-nccl-2node/summary.md` |
| `./analyze-nccl-sharp.py` | `out-nccl-2node-sharp/` | `out-nccl-2node-sharp/summary.md`, and whether SHARP actually engaged |
| `./analyze-megatron.py` | `output-megatron/` | `output-megatron/summary.md` + SVG scaling plot |

The figure of merit is TFLOP/s per GPU (gpu-fryer, Megatron-LM) and bus
bandwidth `busbw` in GB/s (NCCL). `ib_write_bw` results are printed at the end
of each `out-ibwrite/*.out` file.

Background and investigations: `notes.md` (inter-node GPUDirect RDMA cap),
`notes-aicr.md` (comparison with the AICR cluster), `sharp.md`,
`research-b200.md` (published work and research directions).
