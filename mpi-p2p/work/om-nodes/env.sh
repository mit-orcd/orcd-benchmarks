
#1: c7 or r8 or tsq
#2: module name

module use /orcd/software/community/001/spack/stage/lauren/modulefiles
module use /orcd/software/community/001/spack/modulefiles/linux-rocky8-x86_64/gcc/12.2.0
module load gcc/12.2.0-x86_64 
module load openmpi/$2

module list

which mpirun

test_dir="$1-$2"

INSTALL_ROOT=/orcd/home/001/shaohao/mpi/osu-bench/install
INSTALL_DIR=${INSTALL_ROOT}/${test_dir}
OSU_BENCH_HOME=${INSTALL_DIR}/libexec/osu-micro-benchmarks/mpi
export PATH=$PATH:${OSU_BENCH_HOME}/pt2pt:${OSU_BENCH_HOME}/collective:${OSU_BENCH_HOME}/one-sided:${OSU_BENCH_HOME}/startup

which osu_bw

