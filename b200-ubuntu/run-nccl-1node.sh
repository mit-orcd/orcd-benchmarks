#!/bin/bash
# NCCL collective perf on a single B200 node, all GPUs (intra-node NVLink).
# Ubuntu nodes (node5700/node5701): no Slurm, so ssh to the node first, then
# ./run-nccl-1node.sh
#
# Usage: ./run-nccl-1node.sh [collectives] [ngpus]
#   collectives: comma/space separated list to select which benchmarks to run,
#                or "all" for every collective. Default: sendrecv
#     names: sendrecv allreduce allgather reducescatter reduce broadcast
#            alltoall gather scatter hypercube   (underscores/dashes ok)
#   ngpus: number of GPUs to use (default: auto-detect all on the node).
#          Can also be set via the NGPUS env var; the positional arg wins.
#
# Examples:
#   ./run-nccl-1node.sh                      # sendrecv only, all GPUs (default)
#   ./run-nccl-1node.sh allreduce            # allreduce only, all GPUs
#   ./run-nccl-1node.sh sendrecv,allreduce   # two collectives, all GPUs
#   ./run-nccl-1node.sh all                  # every collective, all GPUs
#   ./run-nccl-1node.sh allreduce 4          # allreduce on 4 GPUs
#
# Differences from ../b200-nodes/run-nccl-1node.sh (Rocky 8 + Slurm nodes):
#   - no `module load nvhpc/26.1` (no Lmod, /orcd/software not mounted here);
#     the SDK is the local install under ../nvhpc, put on PATH/LD_LIBRARY_PATH.
#   - CUDA 12.9 flavour of NVHPC, not 13.1: the driver here (570.211.01) exposes
#     CUDA 12.8 and CUDA 13 needs r580+. See ../nccl-tests/build-utuntu-nvhpc-26.1.sh.
#   - binaries come from build-utuntu-nvhpc-26.1 (built on Ubuntu); the Rocky
#     build-nvhpc-26.1 binaries cannot run on these nodes.
#   - no scheduler => no exclusivity; refuses to start if the GPUs are busy
#     (override with FORCE=1).

set -u

NCCL_DIR=/orcd/data/orcd/022/benchmarks/nccl-tests
BUILD_DIR=$NCCL_DIR/build-utuntu-nvhpc-26.1
OUT_DIR=$(cd "$(dirname "$0")" && pwd)/out-nccl-1node
mkdir -p "$OUT_DIR"

# NVHPC 26.1 installed on the shared filesystem (no modules on these nodes).
# Use the 12.9 CUDA flavour throughout: cuda + HPC-X OpenMPI + NCCL.
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
   if [ -n "${BIN[$key]:-}" ]; then
      PROGRAMS+=("${BIN[$key]}")
   else
      echo "Unknown collective: '$tok' (known: ${!BIN[*]} all)" >&2
      exit 1
   fi
done

# number of GPUs to use on the node (one MPI task drives all of them via NCCL);
# 2nd positional arg wins, else NGPUS env var, else auto-detect all GPUs
NGPUS="${2:-${NGPUS:-$(nvidia-smi -L | wc -l)}}"

# exclusivity check — no scheduler on these nodes
BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name \
                  --format=csv,noheader 2>/dev/null)
if [ -n "$BUSY" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "GPUs on $(hostname) are already in use:" >&2
    echo "$BUSY" >&2
    echo "Results would be meaningless. Re-run with FORCE=1 to override." >&2
    exit 1
fi

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4

OUT=$OUT_DIR/nccl-1node-$(hostname)-$(date +%Y%m%d-%H%M%S).out

{
   echo "Node        = $(hostname)"
   echo "Date        = $(date)"
   echo "OS          = $(. /etc/os-release; echo "$PRETTY_NAME") / $(uname -r)"
   echo "mpirun      = $(which mpirun)"
   echo "nvcc        = $(which nvcc)"
   echo "NVHPC       = $NVHPC_HOME (CUDA $CUDA_VER flavour)"
   echo "Bin dir     = $BUILD_DIR"
   echo "num_gpu_per_task = $NGPUS"
   echo "collectives = ${PROGRAMS[*]}"
   echo "sizes       = $MIN_SIZE .. $MAX_SIZE (x$FACTOR)"
} | tee "$OUT"

#export NCCL_DEBUG=INFO

for program in "${PROGRAMS[@]}"
do
   echo "%%%%%%%%% $program %%%%%%%%%%" | tee -a "$OUT"
   mpirun -np 1 --mca btl_openib_warn_no_device_params_found 0 \
      "$BUILD_DIR/$program" -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g "$NGPUS" 2>&1 | tee -a "$OUT"
done

echo "Output written to $OUT"

# Use "mpirun -np 1" to run 1 MPI task with multiple GPUs on one node.
# NCCL does the communication between GPUs on the node with NVLinks or PCIe.
