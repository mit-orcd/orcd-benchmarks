#!/bin/bash
# Submit benchmark jobs with recommended global batch size (GBS = 128 x total_GPUs).
# micro-batch-size stays at 4; gradient accumulation steps = GBS / (micro_bs x DP) = 32.
# Usage: bash submit.sh
# To submit a subset, comment out the lines you don't need.

# ---- L40S (2B-param model) ----
#            nodes  tasks  GPUs/node            GBS
#sbatch -N 1 -n 1 --gpus-per-node=l40s:1  $1 job.sh  128   #  1 GPU  total
#sbatch -N 1 -n 1 --gpus-per-node=l40s:2  $1 job.sh  256   #  2 GPUs total
#sbatch -N 1 -n 1 --gpus-per-node=l40s:4  $1 job.sh  512   #  4 GPUs total
#sbatch -N 2 -n 2 --gpus-per-node=l40s:1  $1 job.sh  256   #  2 GPUs total (2 nodes)
#sbatch -N 2 -n 2 --gpus-per-node=l40s:2  $1 job.sh  512   #  4 GPUs total (2 nodes)
#sbatch -N 2 -n 2 --gpus-per-node=l40s:4  $1 job.sh  1024  #  8 GPUs total (2 nodes)

# ---- H200 (7B-param model) ----
#            nodes  tasks  GPUs/node            GBS
sbatch -N 1 -n 1 --gpus-per-node=h200:1  $1 job.sh  128   #  1 GPU  total
sbatch -N 1 -n 1 --gpus-per-node=h200:2  $1 job.sh  256   #  2 GPUs total
sbatch -N 1 -n 1 --gpus-per-node=h200:4  $1 job.sh  512   #  4 GPUs total
sbatch -N 1 -n 1 --gpus-per-node=h200:8  $1 job.sh  1024  #  8 GPUs total
sbatch -N 2 -n 2 --gpus-per-node=h200:1  $1 job.sh  256   #  2 GPUs total (2 nodes)
sbatch -N 2 -n 2 --gpus-per-node=h200:2  $1 job.sh  512   #  4 GPUs total (2 nodes)
sbatch -N 2 -n 2 --gpus-per-node=h200:4  $1 job.sh  1024  #  8 GPUs total (2 nodes)
