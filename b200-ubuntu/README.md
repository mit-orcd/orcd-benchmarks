# b200-ubuntu

## Introduction

The `../b200-nodes` gpu-fryer and NCCL tests, redone on the two B200 nodes that
ran **Ubuntu 24.04**: node5700 and node5701 (8 × B200 each, NVLink 5 inside a
node, 8 × NDR 400 Gb/s InfiniBand rails between nodes). The goal is a direct
Ubuntu vs Rocky 8 comparison on the same hardware.

When these runs were made (2026-08) the nodes had no Slurm, no Lmod, no
`/orcd/software` mount, and driver 570 (CUDA 12.8), so every script runs locally
after `ssh` and uses its own toolchain. The nodes have since been reinstalled
with Rocky 8 (see `../b200-kimi/README.md`); the results here record the Ubuntu
state.

| Benchmark | Same as `../b200-nodes`? |
|---|---|
| gpu-fryer | Same image and fp32 / bf16 / fp8 × 300 s. Uses `/usr/bin/apptainer` and the NVML library `--nv` injects; refuses to start if another user is on the GPUs (`FORCE=1` overrides) |
| NCCL 1-node / 2-node | Same nccl-tests, sizes and collectives, but a CUDA 12.9 build of NVHPC 26.1 (`../nvhpc`, `../nccl-tests/build-utuntu-nvhpc-26.1.sh`), since CUDA 13 needs driver r580+. 2-node launches over ssh with `mpirun`, not Slurm |

Megatron-LM was planned (`plan.md`) but not run here.

## Installation

- gpu-fryer: `../gpu-fryer/gpu-fryer_1.1.0.sif`.
- NCCL: NVHPC 26.1 SDK installed under `../nvhpc`, nccl-tests built with
  `../nccl-tests/build-utuntu-nvhpc-26.1.sh`. The run scripts set `PATH` and
  `LD_LIBRARY_PATH` to it; no `module load`.
- 2-node NCCL needs passwordless ssh between the two nodes, both directions
  (`ubuntu-nccl.md` lists what the admins must set up).

## Usage

No Slurm on these nodes, so there is no automated many-run driver; each script
is one run, started on the node after `ssh`.

### Single runs — scripts in this dir

```bash
ssh node5700
./run-gpu-fryer.sh [seconds]                               # default 300
./run-nccl-1node.sh [collectives] [ngpus]                  # default sendrecv, all GPUs
./run-nccl-2node.sh [collectives] [gpus_per_node] [nodes]  # default sendrecv, 1, node5700,node5701
```

Collectives: `sendrecv allreduce allgather reducescatter reduce broadcast
alltoall gather scatter`, a comma list, or `all`.

Output goes to `out-gpu-fryer/`, `out-nccl-1node/`, `out-nccl-2node/`.

## Analysis

```bash
./analyze-gpu-fryer.py      # -> out-gpu-fryer/summary.md, vs the AICR B200 reference
./analyze-nccl-1node.py     # -> out-nccl-1node/summary.md
./analyze-nccl-2node.py     # -> out-nccl-2node/summary.md, incl. Ubuntu vs Rocky 8
```

Figure of merit: TFLOP/s per GPU (gpu-fryer), bus bandwidth `busbw` in GB/s
(NCCL). `ubuntu-nccl.md` covers what 2-node NCCL needs on these nodes;
`admin-nccl-notes.md` is the follow-up test list for the admins.
