#!/bin/bash
# NCCL all_reduce SHARP test using the AICR cluster's SHARP environment recipe.
#
# Follows the AICR SHARP module (see the "env set up of sharp on aicr" section of
# sharp.md) — SHARP_HOME / NCCL_PLUGIN_HOME / NCCL_HOME / CUDA_HOME, the sharp and
# plugin lib dirs on LD_LIBRARY_PATH, the libnuma LD_PRELOAD, and the scoped
# NCCL_ALGO — adapted to this cluster.
#
# DIFFERENCES FROM THE AICR MODULE, and why:
#   * module: AICR uses nvhpc/26.3, which is NOT installed here (this cluster has
#     23.3 / 24.3 / 24.5 / 26.1). We use nvhpc/26.1, which ships the *same*
#     HPC-X 2.25.1 — so the SHARP runtime and the NCCL SHARP plugin are the same
#     versions AICR runs. Only the CUDA/NCCL wrapper version differs.
#   * NCCL_IB_HCA: AICR sets "^mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_12" (an
#     EXCLUSION list picking their SHARP-connected NICs). That list must NOT be
#     copied here: on these nodes mlx5_7/8/9/10 ARE four of the eight NDR rails,
#     so applying it would discard half the fabric. We default to this cluster's
#     8 NDR rails and expose the list as an argument so variants are easy to try.
#
# Usage: sbatch -w <nodeA>,<nodeB> job-nccl-2node-sharp-aicr.sh [gpus_per_node] [ib_hca]
#   gpus_per_node : default 8 (SHARP needs the full 8 NICs/node to pay off)
#   ib_hca        : NCCL_IB_HCA value; default = this cluster's 8 NDR rails.
#                   Pass an exclusion list (e.g. "^mlx5_0,mlx5_1") to experiment.
#
# The job first runs sharp_hello, a direct probe of the InfiniBand Aggregation
# Manager that does not involve NCCL at all. If no sharp_am is reachable, that is
# reported up front and the SHARP leg below cannot succeed regardless of settings.
#
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=b200:8
#SBATCH --mem=80GB
#SBATCH -t 45
#SBATCH -J nccl-sharp-aicr
#SBATCH --exclusive
#SBATCH -o out-nccl-2node-sharp/%x-%J.out

NCCL_DIR=/orcd/data/orcd/022/benchmarks/nccl-tests
BUILD_DIR=$NCCL_DIR/build-nvhpc-26.1
mkdir -p out-nccl-2node-sharp

module purge
module load nvhpc/26.1          # AICR uses 26.3; not installed here. Same HPC-X 2.25.1.

NVHPC=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
HPCX=$NVHPC/comm_libs/13.1/hpcx/hpcx-2.25.1

# ---- AICR module recipe, mapped onto this cluster's tree --------------------
export SHARP_HOME=$HPCX/sharp
export NCCL_PLUGIN_HOME=$HPCX/nccl_rdma_sharp_plugin
export CUDA_HOME=$NVHPC/cuda
export NCCL_HOME=$NVHPC/comm_libs/nccl
export LD_LIBRARY_PATH=$SHARP_HOME/lib:$NCCL_PLUGIN_HOME/lib:$NCCL_HOME/lib:$CUDA_HOME/lib64:$HPCX/ompi/lib:$LD_LIBRARY_PATH
# AICR prepends libnuma; the SHARP collectives library needs its symbols resolved
# ahead of anything else that may provide them.
export LD_PRELOAD=/lib64/libnuma.so.1

GPUS_PER_TASK="${1:-8}"
# This cluster's 8 NDR (400 Gb/s) rails — NOT AICR's exclusion list (see header).
IB_HCA="${2:-mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15}"

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4

# MPI is only used to exchange the NCCL unique id; HPC-X UCC/UCX crash in
# MPI_Init here, so force that tiny bootstrap onto TCP. NCCL still uses IB.
MPI_FLAGS="--mca pml ob1 --mca btl tcp,self \
   --mca coll_ucc_enable 0 --mca coll_hcoll_enable 0 \
   --mca btl_openib_warn_no_device_params_found 0"

COMMON_ENV="-x LD_LIBRARY_PATH -x LD_PRELOAD \
   -x SHARP_HOME -x NCCL_PLUGIN_HOME -x NCCL_HOME -x CUDA_HOME \
   -x NCCL_IB_DISABLE=0 \
   -x NCCL_NET_GDR_LEVEL=2 \
   -x NCCL_IB_HCA=$IB_HCA \
   -x NCCL_SOCKET_IFNAME=^lo,docker \
   -x NCCL_DEBUG=INFO \
   -x NCCL_DEBUG_SUBSYS=INIT,ENV,NET"

echo "nodes            = $SLURM_JOB_NODELIST"
echo "num_mpi_tasks    = $SLURM_NTASKS"
echo "num_gpu_per_task = $GPUS_PER_TASK"
echo "NCCL_IB_HCA      = $IB_HCA"
echo "SHARP_HOME       = $SHARP_HOME"
echo "NCCL_PLUGIN_HOME = $NCCL_PLUGIN_HOME"

# ---- Probe: is an Aggregation Manager reachable at all? ---------------------
# sharp_hello talks to the fabric's sharp_am directly, with no NCCL involved. It
# is the cleanest yes/no on whether SHARP can work on this subnet.
echo "%%%%% MODE probe %%%%%"
first_hca=$(echo "$IB_HCA" | tr -d '^' | cut -d, -f1)
echo "--- sharp_hello on ${first_hca}:1 ---"
SHARP_COLL_ENABLE_SAT=1 "$SHARP_HOME/bin/sharp_hello" -d "${first_hca}:1" 2>&1 | head -20
echo "--- sharp_cmd topology query ---"
"$SHARP_HOME/bin/sharp_cmd" topology --ib-dev "${first_hca}:1" 2>&1 | head -15

# ---- Leg A: Ring baseline --------------------------------------------------
echo "%%%%% MODE ring %%%%%"
mpirun -np $SLURM_NTASKS $MPI_FLAGS $COMMON_ENV \
   -x NCCL_COLLNET_ENABLE=0 \
   $BUILD_DIR/all_reduce_perf -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK

# ---- Leg B: SHARP on, using the AICR variable set --------------------------
# NCCL_ALGO uses AICR's scoped form: force CollNet for allreduce only, leaving
# every other collective on NCCL's default algorithm selection.
echo "%%%%% MODE sharp %%%%%"
mpirun -np $SLURM_NTASKS $MPI_FLAGS $COMMON_ENV \
   -x NCCL_COLLNET_ENABLE=1 \
   -x SHARP_COLL_LOCK_ON_COMM_INIT=1 \
   -x NCCL_ALGO=allreduce:collnetchain,collnetdirect \
   $BUILD_DIR/all_reduce_perf -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK

echo "%%%%% MODE end %%%%%"
