#!/bin/bash
#SBATCH -p ou_sloan_teaching
#SBATCH -J sloan_teaching
#SBATCH -t 100
#SBATCH --mem=100GB 
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:1
#SBATCH -w node2643   # node[2643-2644] 

#SBATCH -o out.%x-%N-%J

hostname

module load  apptainer/1.1.7-x86_64  squashfuse/0.1.104-x86_64

module list
which singularity

echo "========== 1 GPU ==========" 
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-OOC-1GPU.txt #HPL-1GPU.dat

