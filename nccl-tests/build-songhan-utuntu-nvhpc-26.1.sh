#!/bin/bash
# Build nccl-tests on the *Ubuntu* DGX H100 nodes (node1800 et al.) against the
# locally-installed NVHPC 26.1 SDK. Produces build-songhan-utuntu-nvhpc-26.1/.
#
# Modeled on build-utuntu-nvhpc-26.1.sh (the B200 Ubuntu build); that script and
# its build dir are left untouched. Differences:
#
#   - **H100 gencode (sm_90)**, not B200 sm_100. compute_90 PTX is kept for
#     forward compatibility.
#
#   - Same NVHPC 26.1 / CUDA 12.9 flavour: these nodes have driver 590.48.01
#     (CUDA 13 capable), but the 12.9 flavour is used consistently for cuda +
#     hpcx + nccl so the binaries match the runtime env in
#     ../dgx-nodes/env-ubuntu.sh. CUDA minor-version compatibility applies.
#
#   - Must be run ON an Ubuntu node (ssh node1800), not on the login node:
#     the login node is RHEL and its glibc/rdma-core differ.
#
# Usage:  ssh node1800 /orcd/data/orcd/022/benchmarks/nccl-tests/build-songhan-utuntu-nvhpc-26.1.sh

set -euo pipefail

SRC_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=$SRC_DIR/build-songhan-utuntu-nvhpc-26.1

NVHPC_HOME=/orcd/data/orcd/022/benchmarks/nvhpc/Linux_x86_64/26.1
CUDA_VER=12.9

MPI_HOME=$NVHPC_HOME/comm_libs/$CUDA_VER/hpcx/latest/ompi   # HPC-X OpenMPI
CUDA_HOME=$NVHPC_HOME/cuda/$CUDA_VER
NCCL_HOME=$NVHPC_HOME/comm_libs/$CUDA_VER/nccl

for f in "$MPI_HOME/include/mpi.h" "$CUDA_HOME/bin/nvcc" "$NCCL_HOME/include/nccl.h"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

export PATH=$CUDA_HOME/bin:$MPI_HOME/bin:$NVHPC_HOME/compilers/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo "node       = $(hostname)  ($(. /etc/os-release; echo "$PRETTY_NAME"))"
echo "kernel     = $(uname -r)"
echo "nvcc       = $(command -v nvcc)"
echo "mpirun     = $(command -v mpirun)"
echo "MPI_HOME   = $MPI_HOME"
echo "CUDA_HOME  = $CUDA_HOME"
echo "NCCL_HOME  = $NCCL_HOME"
echo "BUILD_DIR  = $BUILD_DIR"
echo "verbs      = $(ibv_devinfo -l 2>/dev/null | head -1)"

# Hopper (H100 = sm_90); keep compute_90 PTX for forward compat
GENCODE="-gencode=arch=compute_90,code=sm_90 \
         -gencode=arch=compute_90,code=compute_90"

rm -rf "$BUILD_DIR"
cd "$SRC_DIR"
make -j 16 MPI=1 \
     MPI_HOME="$MPI_HOME" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
     NVCC_GENCODE="$GENCODE" BUILDDIR="$BUILD_DIR"

echo
echo "Built binaries in $BUILD_DIR:"
ls "$BUILD_DIR" | grep -E '_perf$' || true
echo
echo "Link check (all libs must resolve with the env exported above):"
ldd "$BUILD_DIR/all_reduce_perf" | grep -E "libmpi|libnccl|libcudart|not found" || true
