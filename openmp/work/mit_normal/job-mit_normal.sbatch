#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -C rocky8
#SBATCH -t 60
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -w node1600
#SBATCH -o out_files/full-%N-%J.out
#SBATCH --exclusive
# #SBATCH --reservation=monthly_maint

hostname
echo "Requested cores = $SLURM_NTASKS"

for i in 1 2 4 8 16 48 96 144 192
do
  export OMP_NUM_THREADS=$i
  echo "====== Run with $OMP_NUM_THREADS threads. ======"
  time /orcd/data/orcd/001/benchmarks/openmp/src/mat_mul
done

