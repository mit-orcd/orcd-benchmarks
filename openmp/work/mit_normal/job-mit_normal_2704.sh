#!/bin/bash
#SBATCH -J high-l3
#SBATCH -t 10
#SBATCH -n 32 # 8 # 16 # 32 # 64
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --partition=mit_normal
#SBATCH --constraint=high_l3
#SBATCH --reservation=cache_test
#SBATCH -o output/out.%x-%N-%J

for i in 1 2 4 8 16 32 64
do 
     export OMP_NUM_THREADS=$i
     echo "Ran with OMP_NUM_THREADS=$OMP_NUM_THREADS"
     time ../../src/pi_omp
done


