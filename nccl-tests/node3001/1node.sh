#!/bin/bash
#SBATCH -J ubuntu
#SBATCH -p mit_normal_gpu
#SBATCH -q unlimited
#SBATCH -t 100
#SBATCH -N 1 
#SBATCH -n 8 
#SBATCH --mem=500GB 
#SBATCH --gres=gpu:h200:8
#SBATCH --reservation=orcd_testing
#SBATCH -w node3001

#SBATCH -J nvhpc-24.5
#SBATCH -o out-1node/%x-%N-%J

job_name=$SLURM_JOB_NAME
BUILD_DIR=../build-$job_name

module load nvhpc/24.5
#module load openmpi5.0.6
#module load nvhpc/23.3
module load openmpi/4.1.4

mpirun hostname
which mpirun
which nvcc
echo "Bin dir = $BUILD_DIR"

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4
GPUS_PER_TASK=$1  # 8  # 4

echo "num_cpu = num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task = $GPUS_PER_TASK"

#export NCCL_DEBUG=INFO

for program in sendrecv_perf reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np 1 --mca btl_openib_warn_no_device_params_found 0 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done


# Use "mpirun -np 1" to run 1 MPI task with multiple GPUs on one node. 
# NCCL does the communication between GPUs on the node with NVLinks or PCIe
