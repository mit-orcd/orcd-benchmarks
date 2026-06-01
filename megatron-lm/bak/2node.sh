#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -q unlimited
#SBATCH --job-name=megatron-2node
#SBATCH -N 2
#SBATCH -n 2 
#SBATCH --mem=200GB
#SBATCH --gpus-per-node=l40s:1  #h200:8  #l40s:4  # h200:8 
#SBATCH -t 05:00:00
#SBATCH -o output/2node.%N-%J

module load apptainer/1.4.2
which apptainer

export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"
export imag_path="$work_path/imag"

N_NODES=$SLURM_NNODES
GPU_TYPE=`echo $SLURM_GPUS_PER_NODE |awk -F : '{print $1}'`
N_GPUS=`echo $SLURM_GPUS_PER_NODE |awk -F : '{print $2}'`
echo "===== Number of nodes = $N_NODES ====="
echo "===== Number of GPUs per node = $N_GPUS ====="
echo "===== Host names: $SLURM_NODELIST ======"
echo "===== GPU TYPE: $GPU_TYPE ======"
srun hostname

# Get IP address
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
echo $nodes
nodes_array=($nodes)
master_node=${nodes_array[0]}
master_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)
echo $master_node_ip

echo "====================================="
# --contain blocks host $HOME mounts that may conflict with internal libraries, 
# --cleanenv to avoid accidentally importing host user-site Python packages 
# use -n 2 so that the apptainer ls lanched once on each node
#srun -n $SLURM_NTASKS apptainer exec \
srun apptainer exec \
    --nv --contain --cleanenv \
    --bind ${megatron_path} \
    "${imag_path}/pytorch_26.02-py3.sif" \
    ${megatron_path}/run-2nodes.sh  $N_NODES  $N_GPUS  $master_node_ip

