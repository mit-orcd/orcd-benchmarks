#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -t 60
#SBATCH -w node1601
#SBATCH -o out_files/adj_timing/%N.out
#SBATCH --exclusive

hostname

module load miniforge/24.3.0-0
conda activate benchmark

for i in 1 2 4 8 16 24 36 48 72 96
do 
  export OMP_NUM_THREADS=$i
  echo "====== Run with $OMP_NUM_THREADS threads. ======"
  time python ../mat_mult.py 
done
