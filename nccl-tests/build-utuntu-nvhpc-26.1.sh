#!/bin/bash
# Build nccl-tests on the *Ubuntu* B200 nodes (node5700/node5701) against the
# locally-installed NVHPC 26.1 SDK. Produces build-utuntu-nvhpc-26.1/.
#
# Modeled on build-nvhpc-26.1.sh (Rocky 8 nodes); neither that script nor any
# existing build dir is modified. Differences from the Rocky 8 build:
#
#   - No Lmod on these nodes and /orcd/software is not mounted, so there is no
#     `module load nvhpc/26.1`. NVHPC lives on the shared filesystem instead:
#     /orcd/data/orcd/022/benchmarks/nvhpc (installed from the NVIDIA tarball).
#
#   - **CUDA 12.9, not 13.1.** The NVHPC cuda_multi bundle ships both. The GPU
#     driver on these nodes is 570.211.01, which exposes CUDA 12.8
#     (`nvaccelinfo` -> "CUDA Driver Version: 12080"); CUDA 13.x requires an
#     r580+ driver. CUDA minor-version compatibility means a 12.9 build runs
#     fine on the 12.8 driver, so the whole 12.9 flavour (cuda + hpcx + nccl)
#     is used consistently. NVHPC's own default symlinks agree
#     (comm_libs/nccl -> 12.9/nccl, cuda/bin -> cuda/12.9/bin).
#
#   - Builds straight into BUILDDIR instead of `make && mv build ...`, so a
#     concurrent build in ./build is never disturbed.
#
# Usage: ./build-utuntu-nvhpc-26.1.sh

set -euo pipefail

SRC_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=$SRC_DIR/build-utuntu-nvhpc-26.1

NVHPC_HOME=/orcd/data/orcd/022/benchmarks/nvhpc/Linux_x86_64/26.1
CUDA_VER=12.9                       # see note above: driver 570 -> CUDA 12.8

MPI_HOME=$NVHPC_HOME/comm_libs/$CUDA_VER/hpcx/latest/ompi   # HPC-X OpenMPI 4.1.9
CUDA_HOME=$NVHPC_HOME/cuda/$CUDA_VER
NCCL_HOME=$NVHPC_HOME/comm_libs/$CUDA_VER/nccl              # NCCL 2.29.2

for f in "$MPI_HOME/include/mpi.h" "$CUDA_HOME/bin/nvcc" "$NCCL_HOME/include/nccl.h"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

export PATH=$CUDA_HOME/bin:$MPI_HOME/bin:$NVHPC_HOME/compilers/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

echo "node       = $(hostname)  ($(. /etc/os-release; echo "$PRETTY_NAME"))"
echo "nvcc       = $(which nvcc)"
echo "mpirun     = $(which mpirun)"
echo "MPI_HOME   = $MPI_HOME"
echo "CUDA_HOME  = $CUDA_HOME"
echo "NCCL_HOME  = $NCCL_HOME"
echo "BUILD_DIR  = $BUILD_DIR"

# Blackwell (B200 = sm_100); keep compute_100 PTX for forward compat
GENCODE="-gencode=arch=compute_90,code=sm_90 \
         -gencode=arch=compute_100,code=sm_100 \
         -gencode=arch=compute_100,code=compute_100"

rm -rf "$BUILD_DIR"
cd "$SRC_DIR"
make -j "$(nproc)" MPI=1 \
     MPI_HOME="$MPI_HOME" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_HOME" \
     NVCC_GENCODE="$GENCODE" BUILDDIR="$BUILD_DIR"

echo
echo "Built binaries in $BUILD_DIR:"
ls "$BUILD_DIR" | grep -E '_perf$' || true
echo
echo "Link check (all libs must resolve with the env exported above):"
ldd "$BUILD_DIR/all_reduce_perf" | grep -E "libmpi|libnccl|libcudart|not found" || true
