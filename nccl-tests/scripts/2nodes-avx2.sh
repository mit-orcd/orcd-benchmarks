#!/bin/bash
#SBATCH -t 30
#SBATCH -N 2
#SBATCH --mem=80GB
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH -J nvhpc-24.5-ompi-4.1.7-avx2
#SBATCH -o out-2node-2gpu/%x-%N-%J
#SBATCH --exclusive
#SBATCH -p sched_mit_psfc_gpu_r8

job_name=$SLURM_JOB_NAME
BUILD_DIR=../build-$job_name

module load nvhpc/24.5
module load openmpi/4.1.7

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

for program in sendrecv_perf # reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np $SLURM_NTASKS --mca btl_openib_warn_no_device_params_found 0 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done


