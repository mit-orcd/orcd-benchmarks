
#1: c7 or r8
#2: module name, such as 4.1.3-pmi-ucx-x86_64, check it with "module av openmpi"

module use /orcd/software/community/001/spack/stage/lauren/modulefiles
module use /orcd/software/community/001/spack/modulefiles/linux-rocky8-x86_64/gcc/12.2.0
module load gcc/12.2.0-x86_64 

module load openmpi/$2
test_dir="$1-$2"

#module load nvhpc/2024_245/$2
# module load nvhpc/2023_233/nvhpc/$2
#test_dir="$1-nvhpc-$2"

which mpicc

INSTALL_ROOT=../../install
INSTALL_DIR=${INSTALL_ROOT}/${test_dir}
OSU_BENCH_HOME=${INSTALL_DIR}/libexec/osu-micro-benchmarks/mpi
export PATH=$PATH:${OSU_BENCH_HOME}/pt2pt:${OSU_BENCH_HOME}/collective:${OSU_BENCH_HOME}/one-sided:${OSU_BENCH_HOME}/startup

which osu_bw

