
#1: c7 or r8
#2: openmpi module version 

#module use /orcd/software/community/001/modulefiles
#module load gcc/12.2.0-x86_64 
#module load openmpi/$2
module load StdEnv
module load gcc
module load openmpi/$2

which mpicc

test_dir="$1-$2"

INSTALL_ROOT=/orcd/data/orcd/001/benchmarks/mpi-p2p/install
INSTALL_DIR=${INSTALL_ROOT}/${test_dir}
OSU_BENCH_HOME=${INSTALL_DIR}/libexec/osu-micro-benchmarks/mpi
export PATH=$PATH:${OSU_BENCH_HOME}/pt2pt:${OSU_BENCH_HOME}/collective:${OSU_BENCH_HOME}/one-sided:${OSU_BENCH_HOME}/startup

ls $OSU_BENCH_HOME

which osu_bw

