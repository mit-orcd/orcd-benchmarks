#!/bin/bash
#SBATCH -p ou_bcs_low
#SBATCH -t 30
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

#SBATCH -J nvhpc-23.3-ompi3 
#SBATCH -w node[1802,1803]     # node[1802,1803]  node[2802,2803]  node[1702,1703]  node[2702,2703]
# #SBATCH --reservation=mpi_test  # use this only for node[2802,2803]
# #SBATCH --reservation=aug27_maint  # use this only for node[2702,2703]

#SBATCH -o out.%x-%N-%J

job_name=$SLURM_JOB_NAME
BUILD_DIR=../build-$job_name

module purge

module use /software/modulefiles
module load nvhpc/2023_233/nvhpc/23.3

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

export NCCL_DEBUG=INFO

for program in sendrecv_perf #reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np $SLURM_NTASKS --mca btl_openib_warn_no_device_params_found 0 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done


