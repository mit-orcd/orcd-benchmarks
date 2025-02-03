#!/bin/bash
#SBATCH -t 100
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -w node2804
#SBATCH --mem=10GB
#SBATCH -o %N-job_recipt.out

cd /orcd/data/orcd/001/benchmarks/gpu-burn-r8

host=$(hostname)

echo "Running single precision (tensor core)"
./gpu_burn -tc 30 > mit_normal_gpu-1/tc32/mit_normal_gpu-${host}-$SLURM_JOB_ID.out 
echo "Finished single precision tensor core. Output saved in tc32 folder"

echo "Running single precision (standard)"
./gpu_burn 30 > mit_normal_gpu-1/std32/mit_normal_gpu-${host}-$SLURM_JOB_ID.out
echo "Finished single precision gpu burn. Output saved to std32 folder"

echo "Running double precision"
./gpu_burn -d 30 > mit_normal_gpu-1/d64/mit_normal_gpu-${host}-$SLURM_JOB_ID.out
echo "Finished double precision gpu burn. Output saved to d64 folder"

