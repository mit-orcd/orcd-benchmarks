#!/bin/bash
#Title: run gpu_burn on all ou_bcs_low nodes
#Author: justinwz@mit.edu
#Purpose: run tensor core, single precision, double precision gpu_burn on all ou_bcs_nodes 
#To run: Submit this to the slurm scheduler. Adjust out directory if needed to avoid overwriting existing files
#Note: Task is abbreviated! I had partitiontimelimit when I tried to run the normal 300 seconds

nodes=(1702 1703 1802 1803 2702 2703 2802 2803)

for i in ${!nodes[@]}; do
        node=${nodes[i]}
	host=node$node
        sbatch << EOF
#!/bin/bash
#SBATCH -t 100
#SBATCH -p ou_bcs_low
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -w node$node
#SBATCH --mem=10GB
#SBATCH -o %N-job_recipt.out

hostname

cd /orcd/data/orcd/001/benchmarks/gpu-burn-r8

echo "Running tensor core single precision"
./gpu_burn -tc 300 > ou_bcs_low/tc32/bcs-${host}-${SLURM_JOB_ID}.out
echo "Finished tensor core gpu burn. Output saved to tc32 folder"

echo "Running single precision (standard)"
./gpu_burn 300 > ou_bcs_low/std32/bcs-${host}-${SLURM_JOB_ID}.out
echo "Finished single precision gpu burn. Output saved to std32 folder"

echo "Running double precision"
./gpu_burn -d 300 > ou_bcs_low/d64/bcs-${host}-${SLURM_JOB_ID}.out
echo "Finished double precision gpu burn. Output saved to d64 folder"

EOF

done
