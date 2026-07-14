#!/bin/bash
# Build nccl-tests against the nvhpc/26.1 stack (compilers + CUDA 13 + bundled
# HPC-X OpenMPI 4.1.9 + NCCL). No separate openmpi module is loaded — the MPI
# comes from nvhpc's comm_libs/hpcx. Produces build-nvhpc-26.1/ for B200 (sm_100).
# Modeled on build.sh; does not modify it.

BUILD_DIR=build-nvhpc-26.1
rm -rf $BUILD_DIR

module purge
module load nvhpc/26.1
which mpirun
which nvcc

NVHPC_HOME=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
# HPC-X OpenMPI bundled with nvhpc/26.1 (mpi.h in include/, libmpi.so.40 in lib/)
MPI_HOME=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi
CUDA_HOME=$NVHPC_HOME/cuda
NCCL_HOME=$NVHPC_HOME/comm_libs/nccl

# Blackwell (B200 = sm_100); keep compute_100 PTX for forward compat
GENCODE="-gencode=arch=compute_90,code=sm_90 \
         -gencode=arch=compute_100,code=sm_100 \
         -gencode=arch=compute_100,code=compute_100"

make MPI=1 MPI_HOME=$MPI_HOME CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME \
     NVCC_GENCODE="$GENCODE"

mv build $BUILD_DIR
echo "Built binaries in $BUILD_DIR"
