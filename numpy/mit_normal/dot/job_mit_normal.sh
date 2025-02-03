#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 192
#SBATCH -t 60
#SBATCH -o out_files/%N-%J.out
#SBATCH --exclusive

hostname

module load miniforge/23.11.0-0

for i in 1 2 4 8 16 96 144 192 288 384
do 
	export OMP_NUM_THREADS=$i
	echo "====== Run with $OMP_NUM_THREADS threads. ======"
	python dot.py
done

