#!/bin/bash
#SBATCH -p pi_cdrennan_gpu
#SBATCH -o out.pi_cdrennan-%J
#SBATCH -w node1805
#SBATCH -N 1
#SBATCH -n 8


module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0-x86_64
module load openmpi/4.1.4-pmi-ucx-x86_64

echo "========================="
hostname
which mpirun
lscpu
echo "========================="

#BIN_DIR=/orcd/home/001/shaohao/mpi/examples/bin-r8
BIN_DIR=/orcd/data/orcd/001/benchmarks/mpi-laplace/bin-r8

$BIN_DIR/laplace_serial < ../inp
echo "========================="
mpirun -np 4 $BIN_DIR/laplace_mpi < ../inp

