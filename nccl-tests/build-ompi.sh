BUID_DIR=build-nvhpc-24.5-ompi-4.1.7-avx2
rm -rf $BUILD_DIR

module load nvhpc/24.5
module load openmpi/4.1.7


NVHPC_HOME=/orcd/software/community/001/rocky8/nvhpc/2024_245/Linux_x86_64/24.5

#MPI_ROOT=/orcd/software/core/001/pkg/openmpi/4.1.7-nvhpc-24.5
#export PATH=$MPI_ROOT/bin:$PATH
#export LD_LIBRARY_PATH=$MPI_ROOT/lib:$LD_LIBRARY_PATH

which mpicc
mpicc --version

make MPI=1 MPI_HOME=/orcd/software/core/001/pkg/openmpi/4.1.7-nvhpc-24.5 CUDA_HOME=$NVHPC_HOME/cuda  NCCL_HOME=$NVHPC_HOME/comm_libs/nccl

mv build $BUID_DIR

