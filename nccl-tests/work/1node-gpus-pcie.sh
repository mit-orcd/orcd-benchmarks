#!/bin/bash
#SBATCH -p sched_mit_hill_cpsfr
#SBATCH -t 100
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2

module purge
module load nvhpc/2023_233/nvhpc/23.3

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4
NUM_GPUS=2

for program in sendrecv_perf reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np 1 --mca btl_openib_warn_no_device_params_found 0 ../build/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $NUM_GPUS
done

