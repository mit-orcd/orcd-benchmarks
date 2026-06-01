#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -q unlimited
#SBATCH --job-name=megatron
#SBATCH -N 1 
#SBATCH -n 1 
#SBATCH --gres=gpu:2   # h200:2
#SBATCH --time=01:00:00
#SBATCH -o output/out.%N-%J

module load apptainer/1.4.2
which apptainer

export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"
export imag_path="$work_path/imag"

# --contain blocks host $HOME mounts that may conflict with internal libraries, --cleanenv to avoid accidentally importing host user-site Python packages 
# srun apptainer exec \
srun -n 1 apptainer exec \
    --nv --contain --cleanenv \
    --bind ${megatron_path} \
    "${imag_path}/pytorch_26.02-py3.sif" \
    ${megatron_path}/run.sh

