#!/bin/bash
#SBATCH -p sched_mit_orcd
#SBATCH -o out.mit_orcd-%J
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -w node1625  # node[1614-1625]
#SBATCH --reservation=orcd_test

#source /etc/profile.d/modules.sh 
module load gcc/12.2.0-x86_64
module load openmpi/4.1.4-pmi-ucx-x86_64

echo "========================="
hostname
which mpirun
lscpu
echo "========================="

BIN_DIR=/orcd/data/orcd/001/benchmarks/mpi-laplace/bin-r8

$BIN_DIR/laplace_serial < inp
echo "========================="
mpirun -np 4 $BIN_DIR/laplace_mpi < inp

