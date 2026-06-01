#!/bin/bash
#SBATCH -p sched_mit_byildiz
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -t 60

hostname

module load miniforge/23.11.0-0

for i in 1 2 4 8 16 28 56 84 112
do 
  export OMP_NUM_THREADS=$i
  echo "====== Run with $OMP_NUM_THREADS threads. ======"
  time python mat_mult.py 
done


