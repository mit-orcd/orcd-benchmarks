#!/bin/bash
#SBATCH -t 30
#SBATCH -N 2
#SBATCH --mem=80GB
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:1
#SBATCH -J nvhpc-24.5-ompi-5.0.6
#SBATCH -o out-2node-2gpu/%x-%N-%J
#SBATCH --exclusive

job_name=$SLURM_JOB_NAME
BUILD_DIR=../build-$job_name

module load nvhpc/24.5
#module load openmpi/5.0.6
#module load openmpi/4.1.4

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

#export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=ib5
#export NCCL_IB_HCA=mlx5_5

echo $NCCL_DEBUG
echo $NCCL_SOCKET_IFNAME
echo $NCCL_IB_HCA

for program in sendrecv_perf # reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np $SLURM_NTASKS --mca btl_openib_warn_no_device_params_found 0 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done


