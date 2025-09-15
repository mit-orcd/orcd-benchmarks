#!/bin/bash
#SBATCH -t 30
#SBATCH -p mit_normal #pi_ashia07
#SBATCH -n 64
#SBATCH -N 1
#SBATCH -w node1632
#SBATCH --mem=0
#SBATCH -o out_half.%N-%J

module load deprecated-modules
module use /orcd/software/community/001/old_modulefiles/rocky8
module load gcc/12.2.0-x86_64
module load openmpi/4.1.4-pmi-ucx-x86_64

which mpirun

for j in 1 2 4 8 16 32 64 96 128;
do
     export NUM_THREADS=$j
     echo "Ran with MPI_NUM_THREADS=$NUM_THREADS"
     mpirun --oversubscribe -np ${NUM_THREADS} ../src/calc_pi_mpi_big # oversubscribe keyword allows hyperthreading
done

