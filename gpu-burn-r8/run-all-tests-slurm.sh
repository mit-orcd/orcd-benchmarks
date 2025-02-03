#!/bin/bash
#Title: run gpu_burn on specified nodes
#Author: justinwz@mit.edu
#Purpose: run tensor core, single precision, double precision gpu_burn on all ou_bcs_nodes 
#To run: Submit this to the slurm scheduler. Adjust out directory if needed to avoid overwriting existing files

nodes=(2804 2906) # CHANGE HERE

for i in ${!nodes[@]}; do
        node=${nodes[i]}
	host=node$node
        sbatch << EOF
#!/bin/bash
#SBATCH -t 100
#SBATCH -p mit_normal_gpu #CHANGE HERE
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -w node$node
#SBATCH --mem=10GB
#SBATCH -o %N-job_recipt.out

cd /orcd/data/orcd/001/benchmarks/gpu-burn-r8

mkdir tc32 std32 d64

hostname

#CHANGE OUTPUT FILE NAME

echo "Running single precision (tensor core)"
./gpu_burn -tc 300 > mit_normal_gpu-1/tc32/mit_normal_gpu-${host}-$SLURM_JOB_ID.out
echo "Finished single precision tensor core. Output saved in tc32 folder"

echo "Running single precision (standard)"
./gpu_burn 300 > mit_normal_gpu-1/std32/mit_normal_gpu-${host}-$SLURM_JOB_ID.out
echo "Finished single precision gpu burn. Output saved to std32 folder"

echo "Running double precision"
./gpu_burn -d 300 > mit_normal_gpu-1/d64/mit_normal_gpu-${host}-$SLURM_JOB_ID.out
echo "Finished double precision gpu burn. Output saved to d64 folder"


EOF

done
