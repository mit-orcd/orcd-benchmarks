#!/bin/bash
#SBATCH -J ubuntu
#SBATCH -p mit_normal_gpu
#SBATCH -q unlimited
#SBATCH -t 100
#SBATCH -N 1 
#SBATCH -n 8 
#SBATCH --mem=500GB 
#SBATCH --gres=gpu:h200:8
#SBATCH --reservation=orcd_testing
#SBATCH -w node3001

hostname

#module load  apptainer/1.1.7-x86_64  squashfuse/0.1.104-x86_64
module load apptainer/1.1.9

module list
which singularity

echo "========== 1 GPU ==========" 
singularity exec --nv ../hpc-benchmarks_24.03.sif /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat
echo "========== 2 GPUs ==========" 
singularity exec --nv ../hpc-benchmarks_24.03.sif mpirun -np 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat
echo "========== 4 GPUs ==========" 
singularity exec --nv ../hpc-benchmarks_24.03.sif mpirun -np 4 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat
echo "========== 8 GPUs ==========" 
singularity exec --nv ../hpc-benchmarks_24.03.sif mpirun -np 8 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-8GPUs.dat

