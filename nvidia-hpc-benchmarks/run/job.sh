#!/bin/bash
#SBATCH -t 100
#SBATCH -p ou_bcs_low  #pi_mshoulde
#SBATCH -q normal
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -w node3804
#SBATCH --gres=gpu:a100:8
#SBATCH --reservation=orcd_testing
#SBATCH --exclusive
#SBATCH -o out/%N-%J.out

hostname

#module load  apptainer/1.1.7-x86_64  squashfuse/0.1.104-x86_64
module load apptainer/1.1.9

module list
which singularity

gpu_count=8
root_dir=/orcd/data/orcd/002/benchmarks/nvidia-hpc-benchmarks
script_dir=$root_dir/run/hpc-benchmarks_25.04.sif

# run a single gpu job
echo "======================= 1 GPU ========================="
singularity exec --nv $script_dir /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat

# run 2 gpu job if possible
if [ $gpu_count -ge 2 ]; then
echo "======================= 2 GPUs ========================="
singularity exec --nv $script_dir mpirun -np 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat
fi 

# run 4 gpu job if possible
if [ $gpu_count -ge 4 ]; then
echo "====================== 4 GPUs ========================="
singularity exec --nv $script_dir mpirun -np 4 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat
fi

# run 8 gpu job if possible
if [ $gpu_count -ge 8 ]; then
echo "====================== 8 GPUs ========================="
singularity exec --nv $script_dir mpirun -np 8 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-8GPUs.dat
fi

