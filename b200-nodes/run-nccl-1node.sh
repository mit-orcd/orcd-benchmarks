#!/bin/bash
# NCCL collective perf on a single B200 node, all GPUs (intra-node NVLink).
# Runs locally on the node (no slurm) — ssh to the node first, then ./run-nccl-1node.sh
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
#   ./run-nccl-1node.sh all 2                # every collective on 2 GPUs

NCCL_DIR=/orcd/data/orcd/022/benchmarks/nccl-tests
BUILD_DIR=$NCCL_DIR/build-nvhpc-26.1
OUT_DIR=$(cd "$(dirname "$0")" && pwd)/out-nccl-1node
mkdir -p "$OUT_DIR"

# nvhpc/26.1 provides compilers, CUDA 13, NCCL, and its bundled HPC-X OpenMPI.
# No separate openmpi module is loaded — use the MPI from nvhpc.
module purge
module load nvhpc/26.1

# the bundled HPC-X OpenMPI libs (libmpi.so.40) are not on LD_LIBRARY_PATH by
# default; add the ompi lib dir so the nccl-tests binaries can load them
NVHPC_HOME=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
export LD_LIBRARY_PATH=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi/lib:$LD_LIBRARY_PATH

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

# number of GPUs to use on the node (one MPI task drives all of them via NCCL);
# 2nd positional arg wins, else NGPUS env var, else auto-detect all GPUs
NGPUS="${2:-${NGPUS:-$(nvidia-smi -L | wc -l)}}"

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4

OUT=$OUT_DIR/nccl-1node-$(hostname)-$(date +%Y%m%d-%H%M%S).out

which mpirun                             | tee "$OUT"
which nvcc                               | tee -a "$OUT"
echo "Bin dir = $BUILD_DIR"              | tee -a "$OUT"
echo "num_gpu_per_task = $NGPUS"         | tee -a "$OUT"
echo "collectives = ${PROGRAMS[*]}"      | tee -a "$OUT"

#export NCCL_DEBUG=INFO

for program in "${PROGRAMS[@]}"
do
   echo "%%%%%%%%% $program %%%%%%%%%%" | tee -a "$OUT"
   mpirun -np 1 --mca btl_openib_warn_no_device_params_found 0 \
      $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $NGPUS 2>&1 | tee -a "$OUT"
done

echo "Output written to $OUT"

# Use "mpirun -np 1" to run 1 MPI task with multiple GPUs on one node.
# NCCL does the communication between GPUs on the node with NVLinks or PCIe.
