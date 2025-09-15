#!/bin/bash
#SBATCH -p ou_bcs_low
#SBATCH -t 30
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

#SBATCH -J nvhpc-23.3-ompi3 
#SBATCH -w node[1802,1803]   #  node[1802,1803]  node[2802,2803]  node[1702,1703]
# #SBATCH --reservation=mpi_test

#SBATCH -o out.%x-%N-%J

job_name=nvhpc-23.3-ompi3  
#job_name=nvhpc-23.3-opmi4.1.4
#job_name=nvhpc-24.5-ompi3  
#job_name=nvhpc-24.5-opmi4.1.4

BUILD_DIR=../build-$job_name

module purge

module use /software/modulefiles
module load nvhpc/2023_233/nvhpc/23.3

#module use /orcd/software/community/001/modulefiles/rocky8
#module load  nvhpc/2024_245/24.5

#module load gcc/12.2.0-x86_64
#module load openmpi/4.1.4-pmi-ucx-x86_64

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


