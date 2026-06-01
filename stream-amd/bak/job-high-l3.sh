#!/bin/bash
#SBATCH -J high-l3
#SBATCH -t 10
#SBATCH -n 64 # 8 # 16 # 32 # 64 
#SBATCH -N 1 
#SBATCH --mem=0 
#SBATCH --partition=mit_normal 
#SBATCH --constraint=high_l3 
#SBATCH --reservation=cache_test
#SBATCH -o output/out.%x-%N-%J

hostname
lscpu
echo "-----------------------------------------"
echo "number of cpu cores = $SLURM_CPUS_ON_NODE"

# Optimize OpenMP performance behavious
#export OMP_SCHEDULE=static  # Disable dynamic loop scheduling
#export OMP_PROC_BIND=TRUE   # Bind threads to specific resources. Bad.
#export OMP_DYNAMIC=false    # Disable dynamic thread pool sizing

# OMP_PLACES is used for binding OpenMP threads to cores
# See: https://www.openmp.org/spec-html/5.0/openmpse53.html
# For example, a dual socket AMD 4th Gen EPYC™ Processor with 192 (96x2) cores,
# with 4 threads per L3 cache: 96 total places, stride by 2 cores:
# export OMP_PLACES=0:64:2  # bad

for i in 32 64 128
do
  echo "###############################################################"
  export OMP_NUM_THREADS=$i # $SLURM_CPUS_ON_NODE
  echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"

  STREAM_PATH=/orcd/data/orcd/001/benchmarks/memory-bw/stream-amd/STREAM
  echo "=================================================="
  echo "array size = 100M"
  ${STREAM_PATH}/stream_c-100M-100N
  echo "=================================================="
  echo "array size = 430M"
  ${STREAM_PATH}/stream_c-430M-100N
done 
