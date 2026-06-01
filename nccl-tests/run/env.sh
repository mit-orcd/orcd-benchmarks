
#BUILD_DIR=../build-nvhpc-24.5
#BUILD_DIR=../build-nvhpc-24.5-ompi-4.1.4
BUILD_DIR=../build-nvhpc-24.5-ompi-5.0.8
#BUILD_DIR=../build-nvhpc-24.5-ompi-5.0.8-avx2

module load nvhpc/24.5

#module load openmpi/4.1.4
#export LD_LIBRARY_PATH=/orcd/software/core/001/spack/pkg/openmpi/4.1.4/zuyo6jx/lib:$LD_LIBRARY_PATH

module load openmpi/5.0.8

#MPI_ROOT=/orcd/software/core/001/pkg/openmpi/5.0.8-avx2
#export PATH=$MPI_ROOT/bin:$PATH
#export LD_LIBRARY_PATH=$MPI_ROOT/lib:$LD_LIBRARY_PATH

