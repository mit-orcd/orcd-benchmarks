#!/bin/bash
#SBATCH -p sched_mit_byildiz
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -t 60

hostname

module load miniforge/23.11.0-0

export OMP_NUM_THREADS=1
echo "====== Run with $OMP_NUM_THREADS threads. ======"
time python mat_mult.py 

export OMP_NUM_THREADS=2
echo "====== Run with $OMP_NUM_THREADS threads. ======"
time python mat_mult.py 

export OMP_NUM_THREADS=4
echo "====== Run with $OMP_NUM_THREADS threads. ======"
time python mat_mult.py 

export OMP_NUM_THREADS=8
echo "====== Run with $OMP_NUM_THREADS threads. ======"
time python mat_mult.py 

export OMP_NUM_THREADS=28
echo "====== Run with $OMP_NUM_THREADS threads. ======"
time python mat_mult.py 

export OMP_NUM_THREADS=56
echo "====== Run with $OMP_NUM_THREADS threads. ======"
time python mat_mult.py 

export OMP_NUM_THREADS=112
echo "====== Run with $OMP_NUM_THREADS threads. ======"
time python mat_mult.py 


