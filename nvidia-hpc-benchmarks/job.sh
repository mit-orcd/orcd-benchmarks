#!/bin/bash
#SBATCH -t 100
#SBATCH -p sched_mit_psfc_gpu_r8 
#SBATCH -N 1 
#SBATCH -n 8 
#SBATCH --mem=30GB 
#SBATCH --gres=gpu:4 

hostname

module load  apptainer/1.1.7-x86_64  squashfuse/0.1.104-x86_64

module list
which singularity

echo "========== 1 GPU ==========" 
singularity exec --nv $PWD/hpc-benchmarks_24.03.sif /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat
echo "========== 2 GPUs ==========" 
singularity exec --nv $PWD/hpc-benchmarks_24.03.sif mpirun -np 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat
echo "========== 4 GPUs ==========" 
singularity exec --nv $PWD/hpc-benchmarks_24.03.sif mpirun -np 4 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat
#echo "========== 8 GPUs ==========" 
#singularity exec --nv $PWD/hpc-benchmarks_24.03.sif mpirun -np 8 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-8GPUs.dat

