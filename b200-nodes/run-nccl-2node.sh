#!/bin/bash
# NCCL sendrecv_perf across two B200 nodes, one GPU per node (inter-node network).
# Slurm batch job for the mit_testing partition, pinned to node5500 + node5502.
# Submit with:
#     sbatch run-nccl-2node.sh
#SBATCH -p mit_testing
#SBATCH -w node5500,node5502
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=b200:1
#SBATCH --mem=80GB
#SBATCH -t 30
#SBATCH -J nccl-2node-2gpu
#SBATCH --exclusive
#SBATCH -o out-nccl-2node/%x-%J.out

NCCL_DIR=/orcd/data/orcd/022/benchmarks/nccl-tests
BUILD_DIR=$NCCL_DIR/build-nvhpc-26.1
mkdir -p out-nccl-2node

# nvhpc/26.1 provides compilers, CUDA 13, NCCL, and its bundled HPC-X OpenMPI.
module purge
module load nvhpc/26.1

# the bundled HPC-X OpenMPI libs (libmpi.so.40) are not on LD_LIBRARY_PATH by
# default; add the ompi lib dir so the nccl-tests binaries can load them
NVHPC_HOME=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
OMPI_HOME=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi
export LD_LIBRARY_PATH=$OMPI_HOME/lib:$LD_LIBRARY_PATH

GPUS_PER_TASK=1     # one GPU per node, one MPI rank per node

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4

mpirun hostname
which mpirun
echo "Bin dir = $BUILD_DIR"
echo "num_cpu = num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task = $GPUS_PER_TASK"

#export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=ib5
#export NCCL_IB_HCA=mlx5_5

# MPI is used only to exchange the small NCCL unique-id at startup; the GPU data
# path is NCCL, not MPI. The HPC-X UCC/UCX collective components crash in
# MPI_Init here (fatal UCX UD-endpoint timeout bootstrapping over IB), so force
# the tiny MPI bootstrap onto TCP and disable UCC/hcoll. NCCL still uses IB.
MPI_FLAGS="--mca pml ob1 --mca btl tcp,self \
   --mca coll_ucc_enable 0 --mca coll_hcoll_enable 0 \
   --mca btl_openib_warn_no_device_params_found 0"

for program in sendrecv_perf # reduce_perf broadcast_perf all_reduce_perf alltoall_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np $SLURM_NTASKS $MPI_FLAGS \
      $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done

# Two MPI ranks, one per node (Slurm places one task per node), each driving 1
# GPU. NCCL routes the traffic over the inter-node fabric (InfiniBand); busbw
# here measures node-to-node GPU communication, not NVLink.
