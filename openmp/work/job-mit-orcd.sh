#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -w node1602
#SBATCH -C rocky8
#SBATCH -t 60
#SBATCH -N 1
#SBATCH -n 48   # 48  # 96
#SBATCH --cores-per-socket=48
#SBATCH -J orcd
#SBATCH -o out.%x-%N-%J-n96

hostname
echo "Requested cores = $SLURM_NTASKS"

for i in 1 2 4 8 12 16 24 32 48 60 72 96
do
  export OMP_NUM_THREADS=$i
  echo "====== Run with $OMP_NUM_THREADS threads. ======"
  time ../src/pi_omp
done


