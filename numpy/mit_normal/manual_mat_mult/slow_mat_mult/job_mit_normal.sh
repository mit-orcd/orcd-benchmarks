#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 96
#SBATCH -t 60
#SBATCH -o out_files/%N-%J.out
#SBATCH --exclusive

hostname

module load miniforge/23.11.0-0

for i in 1 2 4 8 16 24 36 48 72 96
do 
	export NUM_PROCESSES=$i
	echo "====== Run with $NUM_PROCESSES threads. ======"
	python manual_mat_mult.py $NUM_PROCESSES
done

