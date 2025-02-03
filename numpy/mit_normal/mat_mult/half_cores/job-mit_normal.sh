#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -w node1620
#SBATCH -t 60
#SBATCH -o out_files/new/%N-%J.out
#SBATCH --exclusive

hostname

module load miniforge/23.11.0-0
conda activate benchmark

for i in 1 2 4 8 16 96 144 192 288 392
do 
  export OMP_NUM_THREADS=$i
  echo "====== Run with $OMP_NUM_THREADS threads. ======"
  time python ../mat_mult.py 
done

