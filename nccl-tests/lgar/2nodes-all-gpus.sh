#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 30
#SBATCH -N 2
#SBATCH --ntasks-per-node=4 # 1
#SBATCH --gpus-per-node=4   
#SBATCH --mem-per-node=32G

#SBATCH -J nvhpc-24.5
#SBATCH -o out-2node-full/out.%x-%N-%J

# #SBATCH -w node[2702,2703]    # node[1802,1803]  node[2802,2803]  node[1702,1703]  node[2702,2703]
# #SBATCH --reservation=monthly_maint

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
GPUS_PER_TASK=1  # 4, $SLURM_GPUS_PER_NODE

echo "num_cpu = num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task = $GPUS_PER_TASK"

#export NCCL_DEBUG=INFO

for program in sendrecv_perf reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np $SLURM_NTASKS --mca btl_openib_warn_no_device_params_found 0 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done


