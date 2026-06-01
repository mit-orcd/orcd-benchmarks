#!/bin/bash

nodes=(1702 1703 1802 1803 2702 2703 2802 2803)

for i in ${!nodes[@]}; do 
	node=${nodes[i]}
	sbatch << EOF
#!/bin/bash
#SBATCH -o 1node_out/bcs-1node-%N-%j.out
#SBATCH -t 100
#SBATCH -p ou_bcs_low
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -w node[$node]
#SBATCH --gres=gpu:4

hostname

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


EOF

done
