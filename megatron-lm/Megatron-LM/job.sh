#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -q unlimited
#SBATCH --job-name=megatron
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=100GB
#SBATCH --gpus-per-node=1
#SBATCH -t 03:00:00
#SBATCH -o output/out.%N-%J

# $1 = global batch size (passed from submit.sh)
GBS=${1:?ERROR: global batch size must be passed as first argument to job.sh}

module load apptainer/1.4.2
which apptainer

export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"
export imag_path="$work_path/imag"

N_NODES=$SLURM_NNODES
GPU_TYPE=$(echo $SLURM_GPUS_PER_NODE | awk -F: '{print $1}')
N_GPUS=$(echo $SLURM_GPUS_PER_NODE | awk -F: '{print $2}')
echo "===== Number of nodes = $N_NODES ====="
echo "===== Number of GPUs per node = $N_GPUS ====="
echo "===== Host names: $SLURM_NODELIST ======"
echo "===== GPU TYPE: $GPU_TYPE ======"
echo "===== Global batch size = $GBS ====="
srun hostname

# Get IP address of master node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
master_node=${nodes_array[0]}
master_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)
echo $master_node_ip

echo "====================================="
# --contain blocks host $HOME mounts that may conflict with internal libraries
# --cleanenv avoids importing host user-site Python packages
srun apptainer exec \
    --nv --contain --cleanenv \
    --bind ${megatron_path} \
    --bind /dev/infiniband \
    --bind /sys/class/infiniband \
    --bind /sys/class/infiniband_verbs \
    "${imag_path}/pytorch_26.02-py3.sif" \
    ${megatron_path}/run.sh  $N_NODES  $N_GPUS  $master_node_ip  $GPU_TYPE  $GBS
