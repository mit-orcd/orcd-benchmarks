#!/bin/bash
#SBATCH -p ou_bcs_low
#SBATCH -t 100
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

# #SBATCH -w node[1802,1803]
# #SBATCH -o slurm-nvhpc24.5-N1802-1803-self-%J.out
# #SBATCH -o slurm-nvhpc23.3-N1802-1803-self-%J.out

# #SBATCH -w node[1702,1703]
# #SBATCH -o slurm-nvhpc24.5-N1702-1703-%J.out

#SBATCH --reservation=mpi_test
#SBATCH -w node[2802,2803]
#SBATCH -o slurm-nvhpc24.5-N2802-2803-self%J.out
# #SBATCH -o slurm-nvhpc23.3-N2802-2803-self-%J.out


module purge

#module use /software/modulefiles
#module load nvhpc/2023_233/nvhpc/23.3

module use /orcd/software/community/001/modulefiles/rocky8
module load  nvhpc/2024_245/24.5

#module load gcc/12.2.0-x86_64
#module load openmpi/4.1.4-pmi-ucx-x86_64

#BUILD_DIR=../build-nvhpc-24.5-opmi-4.1.4
#BUILD_DIR=../build-nvhpc-23.3-opmi-4.1.4
#BUILD_DIR=../build-nvhpc-23.3
BUILD_DIR=../build-nvhpc-24.5

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
   #mpirun -npernode 8 --mca btl_openib_warn_no_device_params_found 0 ../build/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g 1
   #for i in `seq 1 4`; do
   #   mpirun -npernode 1 --mca btl_openib_warn_no_device_params_found 0 ./set-ib$i.sh ../build/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g 1 >&out.r8.10-ib-$i&
   #done
done


