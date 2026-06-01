#!/bin/bash
#SBATCH -p ou_orcd_everything
#SBATCH -t 30
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4   #8  #4 
#SBATCH --reservation=orcd_testing
#SBATCH -w node4300


#SBATCH -J nvhpc-24.5-ompi-5.0.6
#SBATCH -o out-1node/out.%x-%N-%J

job_name=$SLURM_JOB_NAME
BUILD_DIR=../build-$job_name

module load nvhpc/24.5
module load openmpi/5.0.6

mpirun hostname
which mpirun
which nvcc
echo "Bin dir = $BUILD_DIR"

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4
GPUS_PER_TASK=4

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
