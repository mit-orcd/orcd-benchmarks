#!/bin/bash
#SBATCH -t 30
#SBATCH -n 48
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --partition=ou_ki_highmem
#SBATCH -o output/out.%x-%N-%J

for i in 1 2 4 8 16 24 36 48 72 96
do
     export OMP_NUM_THREADS=$i
     echo "Ran with OMP_NUM_THREADS=$OMP_NUM_THREADS"
     time ../../../src/mat_mul
done

