#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -q unlimited
#SBATCH -t 30
#SBATCH -N 2
#SBATCH --mem=80GB
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h200:1
#SBATCH -J nvhpc-24.5
#SBATCH -o out-2node-2gpu/%x-%N-%J
#SBATCH --reservation=orcd_testing
#SBATCH -w node[3401,4100]

job_name=$SLURM_JOB_NAME
BUILD_DIR=../build-$job_name

module use /orcd/data/orcd/022/benchmarks/nccl-tests/ubuntu/easybuild/modules/all 
module load nvhpc/24.5
module load OpenMPI

mpirun hostname
/usr/bin/which mpirun
/usr/bin/which nvcc
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
   mpirun -np $SLURM_NTASKS --mca btl_openib_warn_no_device_params_found 1 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done


