#!/bin/bash
# NCCL collective perf across two B200 nodes (inter-node InfiniBand).
# Ubuntu nodes (node5700 + node5701): no Slurm, so mpirun launches the remote
# ranks itself over passwordless ssh. Run this on the first node.
#
# Usage: ./run-nccl-2node.sh [collectives] [gpus_per_node] [nodes]
#   collectives:   comma/space separated list, or "all". Default: sendrecv
#     names: sendrecv allreduce allgather reducescatter reduce broadcast
#            alltoall gather scatter   (underscores/dashes ok)
#   gpus_per_node: GPUs per node = GPUs per MPI rank, one rank per node
#                  (default: 1). Total GPUs = 2 x gpus_per_node.
#   nodes:         comma separated pair (default: node5700,node5701)
#
# Examples:
#   ./run-nccl-2node.sh                      # sendrecv, 1 GPU/node
#   ./run-nccl-2node.sh allreduce            # allreduce, 1 GPU/node
#   ./run-nccl-2node.sh all 8                # every collective, 8 GPUs/node
#
# Differences from ../b200-nodes/job-nccl-2node.sh (Rocky 8 + Slurm):
#   - no Slurm: no #SBATCH block, no $SLURM_NTASKS. mpirun gets an explicit
#     host list and starts the remote orted over ssh, so passwordless ssh
#     between the two nodes is required (both directions).
#   - no `module load`: NVHPC 26.1 comes from the shared install under ../nvhpc,
#     CUDA 12.9 flavour (driver 570 exposes CUDA 12.8; CUDA 13 needs r580+).
#   - PATH/LD_LIBRARY_PATH are forwarded with -x: a non-interactive ssh login
#     on the remote node does not inherit them.
#   - binaries from build-utuntu-nvhpc-26.1 (the Rocky build cannot run here).

set -u

NCCL_DIR=/orcd/data/orcd/022/benchmarks/nccl-tests
BUILD_DIR=$NCCL_DIR/build-utuntu-nvhpc-26.1
OUT_DIR=$(cd "$(dirname "$0")" && pwd)/out-nccl-2node
mkdir -p "$OUT_DIR"

NVHPC_HOME=/orcd/data/orcd/022/benchmarks/nvhpc/Linux_x86_64/26.1
CUDA_VER=12.9
MPI_HOME=$NVHPC_HOME/comm_libs/$CUDA_VER/hpcx/latest/ompi
NCCL_LIB=$NVHPC_HOME/comm_libs/$CUDA_VER/nccl/lib
export PATH=$NVHPC_HOME/cuda/$CUDA_VER/bin:$MPI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_LIB:$NVHPC_HOME/cuda/$CUDA_VER/lib64:${LD_LIBRARY_PATH:-}

if [ ! -x "$BUILD_DIR/sendrecv_perf" ]; then
    echo "nccl-tests not built: $BUILD_DIR" >&2
    echo "Build it first: $NCCL_DIR/build-utuntu-nvhpc-26.1.sh" >&2
    exit 1
fi

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
)
ALL_ORDER="sendrecv allreduce allgather reducescatter reduce broadcast alltoall gather scatter"

SELECTION="${1:-sendrecv}"
PROGRAMS=()
for tok in ${SELECTION//,/ }; do
   key=$(echo "$tok" | tr 'A-Z' 'a-z' | tr -d '_-')
   if [ "$key" = "all" ]; then
      for k in $ALL_ORDER; do PROGRAMS+=("${BIN[$k]}"); done
      continue
   fi
   if [ -n "${BIN[$key]:-}" ]; then
      PROGRAMS+=("${BIN[$key]}")
   else
      echo "Unknown collective: '$tok' (known: ${!BIN[*]} all)" >&2
      exit 1
   fi
done

GPUS_PER_TASK="${2:-1}"                  # GPUs per node = GPUs per MPI rank
NODES="${3:-node5700,node5701}"
NODE_A=${NODES%%,*}
NODE_B=${NODES##*,}
HOSTLIST="$NODE_A:1,$NODE_B:1"           # one MPI rank per node
NTASKS=2

# ssh reachability — mpirun would otherwise hang or fail obscurely
for h in "$NODE_A" "$NODE_B"; do
    if [ "$h" != "$(hostname)" ]; then
        if ! timeout 15 ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$h" true 2>/dev/null; then
            echo "passwordless ssh to $h failed — required for mpirun without Slurm" >&2
            exit 1
        fi
    fi
done

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4

# Inter-node NCCL over the 8 B200 NDR (400 Gb/s) GPU rails. Leaving NIC
# selection to NCCL works at 1 GPU/node but fails to connect at 8 GPU/node on
# these nodes; pin the rails explicitly. WARN keeps output bounded while still
# printing the reason on any failure.
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15
export NCCL_SOCKET_IFNAME=${ETH_IF:-eno3}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}   # NCCL_DEBUG=INFO ./run-nccl-2node.sh to debug
NCCL_XENV="-x NCCL_IB_DISABLE -x NCCL_NET_GDR_LEVEL -x NCCL_IB_HCA \
   -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG"

# MPI is used only to exchange the small NCCL unique-id at startup; the GPU data
# path is NCCL, not MPI. The HPC-X UCC/UCX collective components crash in
# MPI_Init here (fatal UCX UD-endpoint timeout bootstrapping over IB), so force
# the tiny MPI bootstrap onto TCP and disable UCC/hcoll. NCCL still uses IB.
#
# Interface pinning is mandatory on these Ubuntu nodes: /etc/hosts carries the
# Debian/Ubuntu default line "127.0.1.1 <hostname>", so a node resolves its own
# name to loopback. Without the *_if_include flags below, Open MPI advertises
# 127.0.1.1 to the peer and the job hangs in MPI_Init with
# "accepted a TCP connection ... but cannot find a corresponding process entry".
# ETH_IF is the routable inband interface (10.1.x.x) on both nodes.
ETH_IF="${ETH_IF:-eno3}"
MPI_FLAGS="--mca pml ob1 --mca btl tcp,self \
   --mca coll_ucc_enable 0 --mca coll_hcoll_enable 0 \
   --mca btl_openib_warn_no_device_params_found 0 \
   --mca plm_rsh_agent ssh \
   --mca btl_tcp_if_include $ETH_IF --mca oob_tcp_if_include $ETH_IF"
# the remote ssh login is non-interactive: carry the toolchain env across
ENV_X="-x PATH -x LD_LIBRARY_PATH"

OUT=$OUT_DIR/nccl-2node-$NODE_A-$NODE_B-$(date +%Y%m%d-%H%M%S).out

{
   echo "Nodes       = $NODE_A,$NODE_B"
   echo "Date        = $(date)"
   echo "OS          = $(. /etc/os-release; echo "$PRETTY_NAME") / $(uname -r)"
   echo "mpirun      = $(which mpirun)"
   echo "NVHPC       = $NVHPC_HOME (CUDA $CUDA_VER flavour)"
   echo "Bin dir     = $BUILD_DIR"
   echo "num_cpu = num_mpi_tasks = $NTASKS"
   echo "num_gpu_per_task = $GPUS_PER_TASK"
   echo "collectives = ${PROGRAMS[*]}"
   echo "sizes       = $MIN_SIZE .. $MAX_SIZE (x$FACTOR)"
   echo "--- rank placement ---"
   mpirun -np $NTASKS -H "$HOSTLIST" $MPI_FLAGS hostname 2>&1
} | tee "$OUT"

for program in "${PROGRAMS[@]}"
do
   echo "%%%%%%%%% $program %%%%%%%%%%" | tee -a "$OUT"
   mpirun -np $NTASKS -H "$HOSTLIST" $MPI_FLAGS $ENV_X $NCCL_XENV \
      "$BUILD_DIR/$program" -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g "$GPUS_PER_TASK" 2>&1 \
      | tee -a "$OUT"
done

echo "Output written to $OUT"

# Two MPI ranks, one per node, each driving GPUS_PER_TASK GPUs. NCCL routes
# traffic over the inter-node fabric (InfiniBand); busbw here measures
# node-to-node GPU communication, not NVLink.
