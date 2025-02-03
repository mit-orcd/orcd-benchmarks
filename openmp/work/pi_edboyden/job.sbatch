#!/bin/bash
#SBATCH -p pi_edboyden
#SBATCH -t 60
#SBATCH -N 1
#SBATCH -n 96
# #SBATCH -n 48
# #SBATCH -w node2809
#SBATCH -w node2810
#SBATCH -o %N-full.out
# #SBATCH -o %N-half.out
#SBATCH -J %N-full
# #SBATCH -J %N-half
#SBATCH --exclusive

hostname
echo "Requested cores = $SLURM_NTASKS"

# Full:
for i in 1 2 4 8 16 48 72 96 144 192
do
  export OMP_NUM_THREADS=$i
  echo "====== Run with $OMP_NUM_THREADS threads. ======"
  time /orcd/data/orcd/001/benchmarks/openmp/src/mat_mul
done

# Half:
#for i in 1 2 4 8 16 24 36 48 72 96
#do
#  export OMP_NUM_THREADS=$i
#  echo "====== Run with $OMP_NUM_THREADS threads. ======"
#  time /orcd/data/orcd/001/benchmarks/openmp/src/mat_mul
#done

