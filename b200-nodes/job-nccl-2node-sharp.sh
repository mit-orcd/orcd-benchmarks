#!/bin/bash
# NCCL all_reduce across two B200 nodes, A/B: Ring (SHARP off) vs SHARP on.
#
# SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) offloads the
# reduction to the InfiniBand switches, so all_reduce becomes a single pass
# instead of ReduceScatter+AllGather. It applies to reduction collectives
# (all_reduce above all) — not to sendrecv/alltoall/gather, which is why this
# script benchmarks all_reduce only.
#
# Both legs run back-to-back inside ONE allocation, so the comparison is on the
# same nodes, same NICs, same session — the reference's methodology.
#
# Submit with:
#     sbatch -w <nodeA>,<nodeB> job-nccl-2node-sharp.sh [gpus_per_node]
#   gpus_per_node: GPUs per node (default 8; SHARP needs the full 8 NICs/node to
#                  pay off — at low GPU counts it is slower than Ring).
#
# Reference: ~/data022/aicr-benchmarks/Benchmark_WG/nccl-tests/results_b200.md
#   Ring 170 GB/s -> SHARP 357 GB/s (2.2x) at 8 GPU/node x 2 nodes.
#
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=b200:8
#SBATCH --mem=80GB
#SBATCH -t 60
#SBATCH -J nccl-sharp
#SBATCH --exclusive
#SBATCH -o out-nccl-2node-sharp/%x-%J.out

NCCL_DIR=/orcd/data/orcd/022/benchmarks/nccl-tests
BUILD_DIR=$NCCL_DIR/build-nvhpc-26.1
mkdir -p out-nccl-2node-sharp

module purge
module load nvhpc/26.1

NVHPC_HOME=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
HPCX=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1
OMPI_HOME=$HPCX/ompi
# HPC-X ships both the SHARP runtime and the NCCL plugin that exposes it to NCCL
# as a "CollNet"; NCCL auto-discovers libnccl-net.so from LD_LIBRARY_PATH.
SHARP_LIB=$HPCX/sharp/lib
PLUGIN_LIB=$HPCX/nccl_rdma_sharp_plugin/lib
export LD_LIBRARY_PATH=$OMPI_HOME/lib:$PLUGIN_LIB:$SHARP_LIB:$LD_LIBRARY_PATH

GPUS_PER_TASK="${1:-8}"
MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4

# Same MPI bootstrap workaround as job-nccl-2node.sh: HPC-X UCC/UCX crash in
# MPI_Init here, so force the tiny MPI id-exchange onto TCP. NCCL still uses IB.
MPI_FLAGS="--mca pml ob1 --mca btl tcp,self \
   --mca coll_ucc_enable 0 --mca coll_hcoll_enable 0 \
   --mca btl_openib_warn_no_device_params_found 0"

# Shared NCCL settings (the 8 B200 NDR 400 Gb/s rails), identical in both legs.
COMMON_ENV="-x LD_LIBRARY_PATH \
   -x NCCL_IB_DISABLE=0 \
   -x NCCL_NET_GDR_LEVEL=2 \
   -x NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15 \
   -x NCCL_SOCKET_IFNAME=^lo,docker \
   -x NCCL_DEBUG=INFO \
   -x NCCL_DEBUG_SUBSYS=INIT,ENV,NET"

echo "num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task = $GPUS_PER_TASK"
echo "nodes = $SLURM_JOB_NODELIST"
echo "sharp lib = $SHARP_LIB"
echo "nccl net plugin dir = $PLUGIN_LIB"

# ---- Leg A: Ring baseline (CollNet explicitly disabled) -------------------
echo "%%%%% MODE ring %%%%%"
mpirun -np $SLURM_NTASKS $MPI_FLAGS $COMMON_ENV \
   -x NCCL_COLLNET_ENABLE=0 \
   $BUILD_DIR/all_reduce_perf -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK

# ---- Leg B: SHARP on (CollNet algorithms, Simple protocol) ----------------
# If the fabric/switches are not SHARP-enabled, or sharpd is not running, NCCL
# logs the CollNet setup failure and silently falls back to Ring — the analyzer
# checks the INIT/NET debug output to confirm which path actually ran.
echo "%%%%% MODE sharp %%%%%"
mpirun -np $SLURM_NTASKS $MPI_FLAGS $COMMON_ENV \
   -x NCCL_COLLNET_ENABLE=1 \
   -x NCCL_ALGO=CollNetChain,CollNetDirect \
   -x NCCL_PROTO=Simple \
   $BUILD_DIR/all_reduce_perf -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK

echo "%%%%% MODE end %%%%%"
