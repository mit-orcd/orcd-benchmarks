nodes=($1)
partition=$2 #mit_normal_gpu #pi_mghassem  # mit_normal 
res=$3 #orcd_testing  # "" #WareWulf_testing  # orcd_testing
qos=$4
cpu_count=$5   # 8 # 4
gpu_type=$6  #l40s  # h200
gpu_count=$7  

root_dir=/orcd/data/orcd/022/benchmarks/nvidia-hpc-benchmarks
output_dir=$root_dir/$partition/output-$gpu_type
mkdir -p $output_dir
image=$root_dir/run/hpc-benchmarks_25.04.sif


# set optional flags
if [[ "$res" == "none" ]]; then
   submit="sbatch"
else
   submit="sbatch --reservation=$res"
fi

# loop through all nodes
for i in ${!nodes[@]}; do 
	host=node${nodes[$i]}
	echo $host
	$submit << EOF
#!/bin/bash
#SBATCH -o $output_dir/$host-%J.out
#SBATCH -t 100
#SBATCH -p $partition
#SBATCH -q $qos
#SBATCH --mem=100GB
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -w $host
#SBATCH --gres=gpu:$gpu_type:1
#SBATCH -J nvidia-hpc
#SBATCH --array=1-$gpu_count

hostname

module load apptainer/1.1.9

module list
which singularity

./execute-h200.sh $image

EOF

done

# mv slurm*.out std-out
