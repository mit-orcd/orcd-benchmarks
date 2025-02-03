#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -o out.%N-%J
#SBATCH -w node1600
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=00:01:00
#SBATCH --exclusive
#SBATCH --reservation=monthly_maint

module use /orcd/software/community/001/modulefiles
module load gcc
module load openmpi


echo "========================="
hostname
which mpirun
lscpu
echo "========================="

BIN_DIR=/orcd/data/orcd/001/benchmarks/mpi-laplace/bin-r8

$BIN_DIR/laplace_serial < ../../inp
echo "========================="
mpirun -np 4 $BIN_DIR/laplace_mpi < ../../inp

