#!/bin/bash
# NCCL sendrecv_perf on a single B200 node, all GPUs (intra-node NVLink).
# Runs locally on the node (no slurm) — ssh to the node first, then ./run-nccl-1node.sh
# Usage: ./run-nccl-1node.sh [ngpus]   (default: auto-detect all GPUs on the node)

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

# number of GPUs to use on the node (one MPI task drives all of them via NCCL)
NGPUS="${1:-$(nvidia-smi -L | wc -l)}"

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4

OUT=$OUT_DIR/nccl-1node-$(hostname)-$(date +%Y%m%d-%H%M%S).out

which mpirun                     | tee "$OUT"
which nvcc                       | tee -a "$OUT"
echo "Bin dir = $BUILD_DIR"      | tee -a "$OUT"
echo "num_gpu_per_task = $NGPUS" | tee -a "$OUT"

#export NCCL_DEBUG=INFO

for program in sendrecv_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%" | tee -a "$OUT"
   mpirun -np 1 --mca btl_openib_warn_no_device_params_found 0 \
      $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $NGPUS 2>&1 | tee -a "$OUT"
done

echo "Output written to $OUT"

# Use "mpirun -np 1" to run 1 MPI task with multiple GPUs on one node.
# NCCL does the communication between GPUs on the node with NVLinks or PCIe.
