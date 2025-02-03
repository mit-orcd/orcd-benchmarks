#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -o out.%N-%J
#SBATCH -w node1600
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=00:01:00
#SBATCH --exclusive

module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0-x86_64
module load openmpi/4.1.4-pmi-ucx-x86_64


echo "========================="
hostname
which mpirun
lscpu
echo "========================="

BIN_DIR=/orcd/data/orcd/001/benchmarks/mpi-laplace/bin-r8

$BIN_DIR/laplace_serial < ../inp
echo "========================="
mpirun -np 4 $BIN_DIR/laplace_mpi < ../inp

