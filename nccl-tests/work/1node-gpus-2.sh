#!/bin/bash
#SBATCH -p sched_mit_psfc_gpu_r8 
#SBATCH -t 100
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3

module purge
module load nvhpc/2023_233/nvhpc/23.3

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4   # increase the size by a factor
NUM_GPUS=3

for program in sendrecv_perf reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np 1 --mca btl_openib_warn_no_device_params_found 0 ../build/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $NUM_GPUS
done

