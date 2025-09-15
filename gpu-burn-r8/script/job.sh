#!bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:4


cd /orcd/data/orcd/002/benchmarks/gpu-burn-r8

./gpu_burn -tc 300  
./gpu_burn 300
./gpu_burn -d 300

