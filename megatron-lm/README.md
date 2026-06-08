# megatron-lm

## Introduction

Single- and multi-node GPT pre-training throughput using NVIDIA's
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM). It runs a fixed
number of training iterations on mock data (no dataset I/O) and reports
per-iteration time and per-GPU throughput (TFLOP/s), so it measures
end-to-end GPU + NVLink/InfiniBand training performance and multi-GPU
scaling.

## Installation

The upstream source, container image, and Python libs are **not**
committed here (too large). Set them up once (see `notes.apptainer`,
`notes.conda`, `notes.flags` for full details and tuning flags):

```bash
module load apptainer
# 1. Pull the NVIDIA PyTorch container (includes Megatron-Core)
apptainer pull imag/pytorch_25.04-py3.sif docker://nvcr.io/nvidia/pytorch:25.04-py3
# 2. Clone Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
# 3. Copy the site launchers into the cloned tree
cp site-scripts/{submit.sh,run.sh,job.sh} Megatron-LM/
```

Docs: <https://docs.nvidia.com/megatron-core/developer-guide/latest/>.
The launchers live in `site-scripts/` so they can be committed without
the upstream source.

## Usage

`job.sh` is the sbatch script for one configuration; it calls `run.sh`
inside the container (which sets NCCL/InfiniBand env and launches
`pretrain_gpt.py` with `--mock-data`). Global batch size is passed as an
argument (`GBS = 128 × total_GPUs` is the recommended setting).

### Automated, many runs — `submit.sh`

Submits a sweep of GPU counts and node counts (1/2/4/8 GPUs, 1–2 nodes)
for both L40S and H200, each with the matching GBS:

```bash
cd Megatron-LM
bash submit.sh                      # optionally pass extra sbatch flags as $1
```

### Single run — `Megatron-LM/job.sh`

```bash
cd Megatron-LM
sbatch -N 1 -n 1 --gpus-per-node=l40s:4 job.sh 512    # 4 GPUs, GBS 512
```

Outputs land in `Megatron-LM/output/`.

## Analysis

Each job logs per-iteration `elapsed time per iteration (ms)` and, with
`--log-throughput`, the per-GPU `throughput per GPU (TFLOP/s/GPU)`. Read
these from the job's output log in `Megatron-LM/output/`; compare TFLOP/s
across GPU counts to judge scaling (higher is better, near-flat per-GPU
TFLOP/s = good scaling).
