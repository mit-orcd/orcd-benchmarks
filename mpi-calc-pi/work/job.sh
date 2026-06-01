#!/bin/bash
#SBATCH -p pi_melkin
#SBATCH -t 60
#SBATCH -N 1
#SBATCH -n 96 # 48  # 96
#SBATCH -J full # half  # full
#SBATCH -o out.%x-%N-%J

hostname
lscpu

echo "================================"
echo "NTASKS = $SLURM_NTASKS"

module load deprecated-modules
module use /orcd/software/community/001/old_modulefiles/rocky8
module load gcc/12.2.0-x86_64
module load openmpi/4.1.4-pmi-ucx-x86_64
#module load openmpi/5.0.6

which mpirun

#for i in 1 2 4 8 16 24 32 48 72 96 144 192
#for i in 1 2 4 8 16 24 32 48 72 96
for i in 48 72 96
do
     export N_TASKS=$i
     echo "Ran with $N_TASKS MPI TASKS."
     mpirun -np ${N_TASKS} ../../src/calc_pi_mpi_big 
done

