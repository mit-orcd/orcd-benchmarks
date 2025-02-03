#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -t 60
#SBATCH -o out_files/%N-%J.out
#SBATCH --exclusive

hostname

module load miniforge/23.11.0-0
 
python pool.py 

