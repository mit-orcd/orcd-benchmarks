#!/bin/bash
#SBATCH -n 96 #Request 96 tasks (cores)
#SBATCH -w node1602
#SBATCH -N 1 #Request 1 node
#SBATCH -t 0-00:30 #Request runtime of 30 minutes
#SBATCH -p mit_normal #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=4000 #Request 4G of memory per CPU
#SBATCH -o output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e error_%j.txt #redirect errors to error_JOBID.txt



for i in 1 2 4 8 16 48 72 96 144 192
do
	export OMP_NUM_THREADS=$i
	echo "====== Run with $OMP_NUM_THREADS threads. ======"
	./MatrixMultiplication
done
