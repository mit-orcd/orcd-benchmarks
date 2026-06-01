#!/bin/bash
#SBATCH -J nv-hpl
#SBATCH -t 01:30:00
#SBATCH --mem=100GB 
#SBATCH -N 1
#SBATCH -n 4 # 8
#SBATCH --gres=gpu:4  # 8  # 8

#SBATCH -w node3600  # 2433, 2434, 2804, 2906
#SBATCH -p pi_laub

#SBATCH -o output/out.%x-%N-%J
# #SBATCH --reservation=node-test

echo "We don't have to run this test for nodes that have L40s, so don't do this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
hostname
nvidia-smi

#module load  apptainer/1.1.7-x86_64  squashfuse/0.1.104-x86_64
module load apptainer/1.1.9

module list
which singularity

echo "========== 1 GPU ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat
echo "========== 2 GPUs ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif mpirun -np 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat
echo "========== 4 GPUs ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif mpirun -np 4 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat
# There are only 4 GPUs in this partition, so skipping next test
#echo "========== 8 GPUs ==========" 
#singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif mpirun -np 8 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-8GPUs.dat

