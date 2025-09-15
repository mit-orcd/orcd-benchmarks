#!/bin/bash
#SBATCH -p sched_mit_byildiz
#SBATCH -C rocky8
#SBATCH -t 60
#SBATCH -N 1
#SBATCH -n 56 # 28  # 56
# #SBATCH --cores-per-socket=28
#SBATCH -J byildiz
#SBATCH -o out.%x-%N-%J-n56

hostname
echo "Requested cores = $SLURM_NTASKS"

for i in 1 2 4 8 16 28 42 56 84 112
do
  export OMP_NUM_THREADS=$i
  echo "====== Run with $OMP_NUM_THREADS threads. ======"
  time ../src/pi_omp
done


