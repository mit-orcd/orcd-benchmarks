#!/bin/bash
#SBATCH -t 15
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40GB
#SBATCH -J ibtest
#SBATCH --exclusive
cd ${SLURM_SUBMIT_DIR:-.} || exit 1
export MAX_SIZE=1G
source ./env-ubuntu.sh || exit 1
echo "BUILD_DIR = $BUILD_DIR"
echo "--- verbs visible in this job? ---"
ibv_devinfo -d mlx5_0 2>&1 | head -3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=$ETH_IF
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
mpirun -np 2 $MPI_FLAGS \
   -x NCCL_IB_DISABLE -x NCCL_NET_GDR_LEVEL -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
   $BUILD_DIR/sendrecv_perf -b 256M -e 1G -f 4 -g 1
