BUID_DIR=build-nvhpc-24.5
rm -rf $BUILD_DIR

module load nvhpc/24.5
which mpirun

NVHPC_HOME=/orcd/software/community/001/rocky8/nvhpc/2024_245/Linux_x86_64/24.5
make MPI=1 MPI_HOME=$NVHPC_HOME/comm_libs/mpi CUDA_HOME=$NVHPC_HOME/cuda  NCCL_HOME=$NVHPC_HOME/comm_libs/nccl

mv build $BUID_DIR

