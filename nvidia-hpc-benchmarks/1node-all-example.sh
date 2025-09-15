#!/bin/bash

partition=mit_normal_gpu  # ou_sloan_gpu  # mit_normal_gpu
qos=unlimited
#nodes=(2804 2906)
#nodes=(3401 3001 4100)  # h200
#nodes=(1634 2804 3002 3003 3004 3005 3006 3007 3008 3202 3203 3204 3205 3206 3207 3208 3302 3402 3403 3404 3405 3406 3407 3408 3500 3501 3502 3503 3504 3505 3506 3507 3508 3509 3510 3511 3512 4102 4103 4104 4105 4106 4107 4108 4200 4201 4202 4203 4204 4205 4206 4207)  # l40s
#nodes=(2433 2434 3000 3001 3100 3101 3200 3201 3300 3301 3400 3401 4100)  # h200
#nodes=(3400)
nodes=(3200)

gpu_type=h200  # l40s  #h200  # l40s
gpu_count=8  #4    # 8
output_dir=/orcd/data/orcd/002/benchmarks/nvidia-hpc-benchmarks/$partition/output
res=monthly_maint

mkdir -p $output_dir
script_dir=/orcd/data/orcd/002/benchmarks/nvidia-hpc-benchmarks/hpc-benchmarks_24.03.sif

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
