# megatron-lm

## Introduction

Single- and multi-node GPT pre-training throughput using NVIDIA's
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM). It runs a fixed
number of training iterations on mock data (no dataset I/O) and reports
per-iteration time and per-GPU throughput (TFLOP/s), so it measures
end-to-end GPU + NVLink/InfiniBand training performance and multi-GPU
scaling.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/megatron-lm/Megatron-LM
sbatch -N 1 -n 1 --gpus-per-node=l40s:4 job.sh 512     # 4 GPUs, GBS = 128 x 4
# when done: steady-state TFLOP/s per GPU (skip the first 10 warm-up iterations)
grep 'throughput per GPU' output/out.*-<jobid> | tail -n +3 \
    | awk -F'TFLOP/s/GPU\\): ' '{split($2,a," "); s+=a[1]; n++} END {print s/n, "TFLOP/s/GPU"}'
```

## Installation

See [install.md](install.md): pull the `pytorch:26.02-py3` container, clone
Megatron-LM, and copy the launchers from `site-scripts/` into it.

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

Each job writes `Megatron-LM/output/out.<node>-<jobid>`. Every 10 iterations it
logs `elapsed time per iteration (ms)` and `throughput per GPU (TFLOP/s/GPU)`.
Iterations 1 and 10 include warm-up, so average from iteration 20 on:

```bash
cd Megatron-LM/output
for f in out.*; do
  printf "%-28s " $f
  grep 'throughput per GPU' $f | tail -n +3 \
    | awk -F'TFLOP/s/GPU\\): ' '{split($2,a," "); s+=a[1]; n++} END {if(n) printf "%.1f TFLOP/s/GPU\n", s/n; else print "no data"}'
done
```

The header lines of each log (`Number of nodes`, `GPUs per node`, `GPU TYPE`,
`Global batch size`) identify the configuration. Compare per-GPU TFLOP/s across
GPU counts to judge scaling: near-flat per-GPU TFLOP/s means good scaling. The
model differs by GPU type (L40S ~2B, H200 ~7B; see `run.sh`), so do not compare
TFLOP/s across GPU types.
