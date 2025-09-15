nodes=($1)
partition=$2 #mit_normal_gpu #pi_mghassem  # mit_normal 
reservation=$3 #orcd_testing  # "" #WareWulf_testing  # orcd_testing
qos=$4
cpu_count=$5   # 8 # 4
gpu_type=$6  #l40s  # h200
gpu_count=$7  

root_dir=/orcd/data/orcd/002/benchmarks/gpu-burn-r8
out_dir=$root_dir/${partition}/output-$gpu_type
mkdir -p $out_dir

for i in ${!nodes[@]}; do
        node=${nodes[i]}
	host=node$node
        sbatch << EOF
#!/bin/bash
#SBATCH -t 100
#SBATCH -p $partition
#SBATCH -q $qos
#SBATCH -N 1
#SBATCH -n $((2*$gpu_count))
#SBATCH --gres=gpu:$gpu_type:$gpu_count
#SBATCH -w $host
#SBATCH --mem=50GB
#SBATCH --reservation=$reservation
#SBATCH --exclusive

hostname

cd /orcd/data/orcd/002/benchmarks/gpu-burn-r8

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

