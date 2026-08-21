#!/bin/bash
#SBATCH -t 180
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH -J nccl-2node
#SBATCH --exclusive
#
# 2-node NCCL (inter-node) on one pair of Ubuntu DGX H100 nodes.
# Submitted by ./run-nccl.sh, which supplies -p/-q/--gpus-per-node/-w/-o.
#
# Usage (via run-nccl.sh):  sbatch <flags> job-nccl-2node.sh <gpus> [collectives]
#   gpus:        GPUs per node = GPUs per MPI rank, one rank per node (default 1)
#   collectives: comma separated short names, or "all" (default all)
#
# NCCL uses the InfiniBand fabric here (NET/IB, 400 Gb/s NDR rails). Verified
# with NCCL_DEBUG=INFO: "NET/IB : Made virtual device ... speed=400000".

cd ${SLURM_SUBMIT_DIR:-.} || exit 1
export MAX_SIZE=${MAX_SIZE:-16G}
source ./env-ubuntu.sh || exit 1

GPUS_PER_TASK=${1:-1}
select_programs "${2:-all}" || exit 1

echo "nodes               = $SLURM_JOB_NODELIST  ($(. /etc/os-release; echo "$PRETTY_NAME"))"
echo "driver / gpu        = $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1)"
echo "mpirun              = $(command -v mpirun)"
echo "Bin dir             = $BUILD_DIR"
echo "num_cpu = num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task    = $GPUS_PER_TASK"
echo "collectives         = ${PROGRAMS[*]}"

# Inter-node NCCL over IB. NIC selection is left to NCCL, which picks the
# 400 Gb/s rails; pin them with NCCL_IB_HCA if a larger GPUS_PER_TASK fails.
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=$ETH_IF
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}   # INFO to debug a failed connection
NCCL_XENV="-x NCCL_IB_DISABLE -x NCCL_NET_GDR_LEVEL -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG"

for program in "${PROGRAMS[@]}"
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np $SLURM_NTASKS $MPI_FLAGS $NCCL_XENV \
      $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done

# Two MPI ranks, one per node, each driving GPUS_PER_TASK GPUs. busbw here
# measures node-to-node GPU communication, not NVLink.
