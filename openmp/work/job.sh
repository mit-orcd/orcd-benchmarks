#!/bin/bash
#SBATCH -p sched_mit_byildiz
#SBATCH -C rocky8
#SBATCH -t 60
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -J byildiz
#SBATCH -o out.%x-%N-%J

hostname

for i in 1 2 4 8 16 24 32
do
  export OMP_NUM_THREADS=$i
  echo "====== Run with $OMP_NUM_THREADS threads. ======"
  time ../src/pi_omp
done


