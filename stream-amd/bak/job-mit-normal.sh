#!/bin/bash
#SBATCH -J mit-normal
#SBATCH -t 10
#SBATCH -n 48  # 24 # 1  # 48 # 192  # 96
#SBATCH -N 1 
#SBATCH --mem=0 
#SBATCH --partition=mit_normal
#SBATCH -o output/out.%x-%N-%J
#SBATCH -w node1600

hostname
echo "number of cpu cores = $SLURM_CPUS_ON_NODE"
lscpu
echo "==============================="

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
/orcd/software/community/001/rocky8/stream/5.10/2yg/bin/stream_c.exe

