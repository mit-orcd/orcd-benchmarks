#module load cuda/12.4.0-x86_64 nvhpc/2023_233/nvhpc/23.3 openmpi/4.1.4-pmi-ucx-x86_64

module use /software/modulefiles
module use /software/spack/share/spack/lmod/linux-rocky8-x86_64/Core
module purge
module load nvhpc/2023_233/nvhpc/23.3
# CUDA_HOME is set in the modules
# make MPI=1 MPI_HOME=$SPACK_PKG_OPENMPI_ROOT NVHPC_CUDA_HOME=$CUDA_HOME NVHPC_HOME=$NVHPC_ROOT
#make MPI=1 MPI_HOME=/software/nvhpc/2023_233/Linux_x86_64/23.3/comm_libs/mpi NVHPC_CUDA_HOME=/software/nvhpc/2023_233/Linux_x86_64/23.3/cuda  NVHPC_HOME=/software/nvhpc/2023_233/Linux_x86_64/23.3/comm_libs/nccl
make MPI=1 MPI_HOME=/software/nvhpc/2023_233/Linux_x86_64/23.3/comm_libs/mpi CUDA_HOME=/software/nvhpc/2023_233/Linux_x86_64/23.3/cuda  NCCL_HOME=/software/nvhpc/2023_233/Linux_x86_64/23.3/comm_libs/nccl

#module load gcc/12.2.0-x86_64
#module load openmpi/4.1.4-pmi-ucx-x86_64
make MPI=1 MPI_HOME=/orcd/software/community/001/rocky8/openmpi/4.1.4/o6z CUDA_HOME=/software/nvhpc/2023_233/Linux_x86_64/23.3/cuda  NCCL_HOME=/software/nvhpc/2023_233/Linux_x86_64/23.3/comm_libs/nccl


module use /orcd/software/community/001/modulefiles/rocky8
module purge
module load nvhpc/2024_245/24.5
make MPI=1 MPI_HOME=/orcd/software/community/001/rocky8/nvhpc/2024_245/Linux_x86_64/24.5/comm_libs/mpi CUDA_HOME=/orcd/software/community/001/rocky8/nvhpc/2024_245/Linux_x86_64/24.5/cuda  NCCL_HOME=/orcd/software/community/001/rocky8/nvhpc/2024_245/Linux_x86_64/24.5/comm_libs/nccl

# best build. Build on compute nodes not on the head node.
module use /orcd/software/community/001/modulefiles/rocky8
module purge
module load nvhpc/2024_245/24.5
module load openmpi/4.1.4-pmi-ucx-x86_64
make MPI=1 MPI_HOME=/orcd/software/community/001/rocky8/openmpi/4.1.4/o6z CUDA_HOME=/orcd/software/community/001/rocky8/nvhpc/2024_245/Linux_x86_64/24.5/cuda  NCCL_HOME=/orcd/software/community/001/rocky8/nvhpc/2024_245/Linux_x86_64/24.5/comm_libs/nccl
