BUID_DIR=build-nvhpc-24.5-ompi-5.0.6
rm -rf $BUILD_DIR

module load nvhpc/24.5
module load openmpi/5.0.6
which mpirun

NVHPC_HOME=/orcd/software/community/001/rocky8/nvhpc/2024_245/Linux_x86_64/24.5
make MPI=1 MPI_HOME=/orcd/software/community/001/pkg/openmpi/5.0.6 CUDA_HOME=$NVHPC_HOME/cuda  NCCL_HOME=$NVHPC_HOME/comm_libs/nccl
#make MPI=1 MPI_HOME=$NVHPC_HOME/comm_libs/mpi CUDA_HOME=$NVHPC_HOME/cuda  NCCL_HOME=$NVHPC_HOME/comm_libs/nccl

mv build $BUID_DIR

