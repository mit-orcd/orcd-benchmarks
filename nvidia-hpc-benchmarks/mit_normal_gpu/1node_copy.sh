#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -w node2906
#SBATCH -t 100
#SBATCH --mem=100GB 
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:4 

#SBATCH -o out_copy.%x-%N-%J

hostname
nvidia-smi

module load apptainer/1.1.9

module list
which singularity

echo "========== 1 GPU ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat
echo "========== 2 GPUs ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif mpirun -np 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat
echo "========== 4 GPUs ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif mpirun -np 4 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat

