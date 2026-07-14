#!/bin/bash
# NCCL collective perf across two B200 nodes (inter-node network).
# Slurm batch job for the mit_testing partition, pinned to node5500 + node5502.
#
# Submit with:
#     sbatch job-nccl-2node.sh [collectives] [gpus_per_node]
#   collectives: comma/space separated list to select which benchmarks to run,
#                or "all" for every collective. Default: sendrecv
#     names: sendrecv allreduce allgather reducescatter reduce broadcast
#            alltoall gather scatter hypercube   (underscores/dashes ok)
#   gpus_per_node: GPUs to use per node = MPI ranks' GPU count (default: 1).
#                  Total GPUs in the job = 2 nodes x gpus_per_node.
#
# Examples:
#     sbatch job-nccl-2node.sh                     # sendrecv, 1 GPU/node
#     sbatch job-nccl-2node.sh allreduce           # allreduce, 1 GPU/node
#     sbatch job-nccl-2node.sh sendrecv,allreduce  # two collectives
#     sbatch job-nccl-2node.sh all 8               # every collective, 8 GPU/node
#
# The job allocates all 8 GPUs per node (--exclusive); the gpus_per_node arg only
# controls how many NCCL actually uses, so no resubmission is needed to change it.
#SBATCH -p mit_testing
#SBATCH -w node5500,node5502
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=b200:8
#SBATCH --mem=80GB
#SBATCH -t 30
#SBATCH -J nccl-2node
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

# map short collective names -> nccl-tests binary names
declare -A BIN=(
   [sendrecv]=sendrecv_perf
   [allreduce]=all_reduce_perf
   [allgather]=all_gather_perf
   [reducescatter]=reduce_scatter_perf
   [reduce]=reduce_perf
   [broadcast]=broadcast_perf
   [alltoall]=alltoall_perf
   [gather]=gather_perf
   [scatter]=scatter_perf
   [hypercube]=hypercube_perf
)
# order used when "all" is requested
ALL_ORDER="sendrecv allreduce allgather reducescatter reduce broadcast alltoall gather scatter hypercube"

# parse the selection argument (default: sendrecv); accept commas, spaces,
# and normalize by stripping underscores/dashes and lowercasing
SELECTION="${1:-sendrecv}"
PROGRAMS=()
for tok in ${SELECTION//,/ }; do
   key=$(echo "$tok" | tr 'A-Z' 'a-z' | tr -d '_-')
   if [ "$key" = "all" ]; then
      for k in $ALL_ORDER; do PROGRAMS+=("${BIN[$k]}"); done
      continue
   fi
   if [ -n "${BIN[$key]}" ]; then
      PROGRAMS+=("${BIN[$key]}")
   else
      echo "Unknown collective: '$tok' (known: ${!BIN[*]} all)" >&2
      exit 1
   fi
done

GPUS_PER_TASK="${2:-1}"   # GPUs per node = GPUs per MPI rank (one rank per node)

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4

mpirun hostname
which mpirun
echo "Bin dir = $BUILD_DIR"
echo "num_cpu = num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task = $GPUS_PER_TASK"
echo "collectives = ${PROGRAMS[*]}"

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

for program in "${PROGRAMS[@]}"
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np $SLURM_NTASKS $MPI_FLAGS \
      $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done

# Two MPI ranks, one per node (Slurm places one task per node), each driving
# GPUS_PER_TASK GPUs. NCCL routes traffic over the inter-node fabric
# (InfiniBand); busbw here measures node-to-node GPU communication, not NVLink.
