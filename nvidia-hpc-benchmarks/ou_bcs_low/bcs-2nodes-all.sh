#!/bin/bash

nodes=(1702 1802 2702 2802)

for i in ${!nodes[@]}; do
	node1=${nodes[$i]}
	node2=$((node1 + 1))
	sbatch << EOF
#!/bin/bash
#SBATCH -o 2node_out/bcs-2nodes-%j.out
#SBATCH -t 100
#SBATCH -p ou_bcs_low
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=100GB
# #SBATCH -n 16
# #SBATCH --gres=gpu:4

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
echo "========== 8 GPUs =========="
singularity exec --nv $PWD/../hpc-benchmarks_24.03.sif mpirun -np 8 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-8GPUs.dat

EOF
done

