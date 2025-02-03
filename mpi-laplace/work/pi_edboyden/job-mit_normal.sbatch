#!/bin/bash
#SBATCH -p pi_edboyden
#SBATCH -o %N.out
#SBATCH -w node2809
# #SBATCH -w node2810
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=00:01:00

#module use /orcd/software/community/001/modulefiles
#module load gcc/12.2.0-x86_64
#module load openmpi/4.1.4-pmi-ucx-x86_64
module load StdEnv
module load gcc
module load openmpi

echo "========================="
hostname
which mpirun
lscpu
echo "========================="

BIN_DIR=/orcd/data/orcd/001/benchmarks/mpi-laplace/bin-r8

$BIN_DIR/laplace_serial < ../inp
echo "========================="
mpirun -np 4 $BIN_DIR/laplace_mpi < ../inp

