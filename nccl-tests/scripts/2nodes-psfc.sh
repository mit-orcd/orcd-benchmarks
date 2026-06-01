#!/bin/bash
#SBATCH -p sched_mit_psfc_gpu_r8
#SBATCH -t 30
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
# #SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

#SBATCH -J nvhpc-24.5
#SBATCH -o out-2node-full/out.%x-%N-%J

job_name=$SLURM_JOB_NAME
BUILD_DIR=../build-$job_name

module load nvhpc/24.5

mpirun hostname
which mpirun
which nvcc
echo "Bin dir = $BUILD_DIR"

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4
GPUS_PER_TASK=$SLURM_GPUS_PER_NODE

echo "num_cpu = num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task = $GPUS_PER_TASK"

#export NCCL_DEBUG=INFO

for program in sendrecv_perf # reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   #mpirun -npernode 1 --mca btl_openib_warn_no_device_params_found 0 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
   mpirun -npernode 4 --mca btl_openib_warn_no_device_params_found 0 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g 1
   #for i in `seq 1 4`   # run multiple p2p MPI between two GPUs on two nodes
   #do 
   #  echo "======= MPI pair 1 on two nodes ======"
   #  mpirun -npernode 1 --mca btl_openib_warn_no_device_params_found 0 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g 1 &
   #done
done


