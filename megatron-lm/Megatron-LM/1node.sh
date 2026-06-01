#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -q unlimited
#SBATCH --job-name=megatron-1node
#SBATCH -N 1 
#SBATCH -n 1 
#SBATCH --mem=200GB
#SBATCH --gpus-per-node=l40s:2  # h200:8  # l40s:4  # h200:8 
#SBATCH -t 05:00:00
#SBATCH -o output/1node.%N-%J

module load apptainer/1.4.2
which apptainer

export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"
export imag_path="$work_path/imag"

GPU_TYPE=`echo $SLURM_GPUS_PER_NODE |awk -F : '{print $1}'`
N_GPUS=`echo $SLURM_GPUS_PER_NODE |awk -F : '{print $2}'`
echo "===== Number of nodes = $SLURM_NNODES ====="
echo "===== Number of GPUs per node = $N_GPUS ====="
echo "===== Host names: $SLURM_NODELIST ======"
echo "===== GPU TYPE: $GPU_TYPE ======"
srun hostname
echo "====================================="

# --contain blocks host $HOME mounts that may conflict with internal libraries, 
# --cleanenv to avoid accidentally importing host user-site Python packages 
# use -n 1 so that the apptainer ls lanched only once
# srun apptainer exec \
srun -n 1 apptainer exec \
    --nv --contain --cleanenv \
    --bind ${megatron_path} \
    "${imag_path}/pytorch_26.02-py3.sif" \
    ${megatron_path}/run-1node.sh  $N_GPUS


