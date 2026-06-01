#!/bin/bash
#SBATCH -J pi-srmadden
#SBATCH -t 10
#SBATCH -n 128  # 64  
#SBATCH -N 1 
#SBATCH --mem=0 
#SBATCH --partition=pi_srmadden
#SBATCH -o out.%x-%N-%J
# #SBATCH -w node1620

hostname
echo "number of cpu cores = $SLURM_CPUS_ON_NODE"
lscpu
echo "==============================="
/orcd/software/community/001/rocky8/stream/5.10/2yg/bin/stream_c.exe

