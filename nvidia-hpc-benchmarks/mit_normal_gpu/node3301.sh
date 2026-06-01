#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --reservation=orcd_testing
#SBATCH -J nvl-node3301
#SBATCH -t 00:60:00
#SBATCH --mem=100GB 
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:8 
#SBATCH -w node3301  
#SBATCH -q unlimited

#SBATCH -o out.%x-%N-%J

hostname

module load  apptainer/1.1.7-x86_64  squashfuse/0.1.104-x86_64

module list
which singularity

echo "========== 1 GPU ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat
echo "========== 2 GPUs ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif mpirun -np 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat
echo "========== 4 GPUs ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif mpirun -np 4 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat
echo "========== 8 GPUs =========="
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif mpirun -np 8 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-8GPUs.dat
