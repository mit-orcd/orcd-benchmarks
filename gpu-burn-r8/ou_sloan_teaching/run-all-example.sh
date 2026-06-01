#!/bin/bash
#Title: run gpu_burn on all ou_bcs_low nodes
#Author: justinwz@mit.edu
#Purpose: run tensor core, single precision, double precision gpu_burn on all ou_bcs_nodes 
#To run: Submit this to the slurm scheduler. Adjust out directory if needed to avoid overwriting existing files

nodes=(2643 2644)
partition=ou_sloan_teaching
gpu_count=4
out_dir=/orcd/data/orcd/001/benchmarks/gpu-burn-r8/${partition}/output

mkdir -p $out_dir

for i in ${!nodes[@]}; do
        node=${nodes[i]}
	host=node$node
        sbatch << EOF
#!/bin/bash
#SBATCH -t 100
#SBATCH -p $partition
#SBATCH -N 1
#SBATCH -n $((2*$gpu_count))
#SBATCH --gres=gpu:$gpu_count
#SBATCH -w $host
#SBATCH --mem=10GB

hostname

cd /orcd/data/orcd/001/benchmarks/gpu-burn-r8

echo "Running tensor core single precision"
./gpu_burn -tc 300 > ${out_dir}/tc32_${host}-\${SLURM_JOB_ID}.out
echo "Finished tensor core gpu burn. Output saved to tc32 folder"

echo "Running single precision (standard)"
./gpu_burn 300 > ${out_dir}/std32_${host}-\${SLURM_JOB_ID}.out
echo "Finished single precision gpu burn. Output saved to std32 folder"

echo "Running double precision"
./gpu_burn -d 300 > ${out_dir}/d64_${host}-\${SLURM_JOB_ID}.out
echo "Finished double precision gpu burn. Output saved to d64 folder"

EOF

done
