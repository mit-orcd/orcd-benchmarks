nodes=($1)
partition=$2 #mit_normal_gpu #pi_mghassem  # mit_normal 
reservation=$3 #orcd_testing  # "" #WareWulf_testing  # orcd_testing
qos=$4
cpu_count=$5   # 8 # 4
gpu_type=$6  #l40s  # h200
gpu_count=$7  

root_dir=/orcd/data/orcd/002/benchmarks/nvidia-hpc-benchmarks
output_dir=$root_dir/$partition/output-$gpu_type
mkdir -p $output_dir
script_dir=$root_dir/run/hpc-benchmarks_25.04.sif

for i in ${!nodes[@]}; do 
	host=node${nodes[$i]}
	echo $host
	sbatch << EOF
#!/bin/bash
#SBATCH -o $output_dir/$host-%J.out
#SBATCH -t 100
#SBATCH -p $partition
#SBATCH -q $qos
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH -n $((2*gpu_count))
#SBATCH -w $host
#SBATCH --gres=gpu:$gpu_type:$gpu_count
#SBATCH --reservation=$res
#SBATCH --exclusive

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
if [ $gpu_count -ge 4 ]; then
echo "====================== 4 GPUs ========================="
singularity exec --nv $script_dir mpirun -np 4 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat
fi

# run 8 gpu job if possible
if [ $gpu_count -ge 8 ]; then
echo "====================== 8 GPUs ========================="
singularity exec --nv $script_dir mpirun -np 8 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-8GPUs.dat
fi

EOF

done

# mv slurm*.out std-out
