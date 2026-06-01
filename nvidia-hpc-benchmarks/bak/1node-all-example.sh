#!/bin/bash

partition=mit_normal_gpu
nodes=(2804 2906)
gpu_count=4
output_dir=/orcd/data/orcd/001/benchmarks/nvidia-hpc-benchmarks/$partition/output

mkdir -p $output_dir
script_dir=/orcd/data/orcd/001/benchmarks/nvidia-hpc-benchmarks/hpc-benchmarks_24.03.sif

for i in ${!nodes[@]}; do 
	host=node${nodes[$i]}
	echo $host
	sbatch << EOF
#!/bin/bash
#SBATCH -o $output_dir/$host-%J.out
#SBATCH -t 100
#SBATCH -p $partition
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH -n $((2*gpu_count))
#SBATCH -w $host
#SBATCH --gres=gpu:$gpu_count
# #SBATCH --reservation=node-test

hostname

#module load  apptainer/1.1.7-x86_64  squashfuse/0.1.104-x86_64
module load apptainer/1.1.9

module list
which singularity

# run a single gpu job
echo "======================= 1 GPU ========================="
singularity exec --nv $script_dir /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat

# run 2 gpu job if possible
if [ $gpu_count -ge 2 ]; then
echo "======================= 2 GPUs ========================="
singularity exec --nv $script_dir mpirun -np 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat
fi 

# run 4 gpu job if possible
if [ $gpu_count -ge 2 ]; then
echo "====================== 4 GPUs ========================="
singularity exec --nv $script_dir mpirun -np 4 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat
fi

EOF

done
