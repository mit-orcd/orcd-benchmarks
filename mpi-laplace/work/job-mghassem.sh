#!/bin/bash
#SBATCH -p pi_mghassem
#SBATCH -o out.pi_mghassem-%J
#SBATCH -N 1
#SBATCH -n 8

module load gcc/12.2.0-x86_64
module load openmpi/4.1.4-pmi-ucx-x86_64


echo "========================="
hostname
which mpirun
lscpu
echo "========================="

BIN_DIR=/orcd/home/001/shaohao/mpi/examples/bin-r8

$BIN_DIR/laplace_serial < inp
echo "========================="
mpirun -np 4 $BIN_DIR/laplace_mpi < inp

