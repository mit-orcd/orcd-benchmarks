
#1: c7 or r8
#2: module name

module load nvhpc/2024_245/$2

which mpicc

test_dir="$1-nvhpc-$2"

INSTALL_ROOT=/orcd/home/001/shaohao/mpi/osu-bench/install
INSTALL_DIR=${INSTALL_ROOT}/${test_dir}
OSU_BENCH_HOME=${INSTALL_DIR}/libexec/osu-micro-benchmarks/mpi
export PATH=$PATH:${OSU_BENCH_HOME}/pt2pt:${OSU_BENCH_HOME}/collective:${OSU_BENCH_HOME}/one-sided:${OSU_BENCH_HOME}/startup

which osu_bw

