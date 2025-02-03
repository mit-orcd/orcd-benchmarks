#!/bin/bash
#SBATCH -J bcs
#SBATCH -t 10
#SBATCH -n 64
#SBATCH -N 1 
#SBATCH --mem=0 
#SBATCH --partition=ou_bcs_low
#SBATCH -o out.%x-%N-%J

/orcd/software/community/001/rocky8/stream/5.10/2yg/bin/stream_c.exe

