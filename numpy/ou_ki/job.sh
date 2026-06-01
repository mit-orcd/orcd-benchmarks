#!/bin/bash
#SBATCH -t 30
#SBATCH -p ou_ki
#SBATCH -n 48
#SBATCH -N 1
#SBATCH -w node9801
#SBATCH -o out_half.%N-%J

module load miniforge/23.11.0-0

hostname

for j in 1 2 4 8 16 24 36 48 72 96
do
     export OMP_NUM_THREADS=$j
     echo "Ran with OMP_NUM_THREADS=$OMP_NUM_THREADS"
     python mat_mult.py
done
