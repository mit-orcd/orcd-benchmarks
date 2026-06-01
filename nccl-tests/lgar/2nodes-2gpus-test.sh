#!/bin/bash
#SBATCH -t 30
#SBATCH -N 2
#SBATCH --mem=80GB
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1
#SBATCH -J nvhpc-24.5-opmi-4.1.4
#SBATCH --exclusive
#SBATCH --reservation=orcd_testing
#SBATCH --partition=ou_orcd_everything
#SBATCH -w node3009,node3209
#SBATCH -o out-2node-2gpu/last-run
####SBATCH -o out-2node-2gpu/%x-%N-%J


job_name=$SLURM_JOB_NAME
BUILD_DIR=../build-$job_name


module load nvhpc/24.5
module load openmpi/4.1.4

mpirun hostname
which mpirun
which nvcc
echo "Bin dir = $BUILD_DIR"

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4
GPUS_PER_TASK=1  #  $(echo $GPUS_PER_TASK | awk -F\: '{print $2}') # $SLURM_GPUS_PER_NODE

echo "num_cpu = num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task = $GPUS_PER_TASK"

# See what NCCL/UCX are doing
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# 1) Make NCCL bootstrap over IPoIB sockets (not eth/containers)
export NCCL_SOCKET_IFNAME='ib*'     # or exclude others: '^lo,eth*,en*,br*,docker*,veth*'

# 2) Constrain RDMA devices to your good rails (port 1 shown)
export NCCL_IB_HCA='mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1'
export NCCL_CROSS_NIC=1

# 3) Force pure InfiniBand GIDs (avoid accidental RoCE indices)
export NCCL_IB_GID_INDEX=0
# If RoCE exists anywhere, also do:
export UCX_IB_GID_INDEX=0

# 4) Use UCX with OMPI and pin its net devices to the same rails
export UCX_TLS=rc,cuda_copy,cuda_ipc,sm,self
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
# Optional while testing to avoid MTU surprises:
# export NCCL_IB_MTU=1024 ; export UCX_IB_MTU=1024

mpirun -np $SLURM_NTASKS \
  --mca pml ucx --mca osc ucx \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_CROSS_NIC -x NCCL_IB_GID_INDEX \
  -x UCX_TLS -x UCX_NET_DEVICES -x UCX_IB_GID_INDEX -x UCX_IB_MTU \
  $BUILD_DIR/sendrecv_perf -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
