#!/bin/bash
#Title: run gpu_burn on a specified partition
#Author: justinwz@mit.edu
#Purpose: run tensor core, single precision, double precision gpu_burn on all ou_bcs_nodes 
#To run: Submit this to the slurm scheduler. Adjust out directory if needed to avoid overwriting existing files

#nodes=(4401 4402 4403 4404)
#nodes=(4208 4209 4210 4211 4212 4302 4303 4304 4305 4502 4503 4504)
#nodes=(4104 4207)
#nodes=(3401 3001 4100)
#nodes=(4306 4307 4308)
nodes=(4505 4506 4507)
#nodes=(1634 2804 3002 3003 3004 3005 3006 3007 3008 3202 3203 3204 3205 3206 3207 3208 3302 3402 3403 3404 3405 3406 3407 3408 3500 3501 3502 3503 3504 3505 3506 3507 3508 3509 3510 3511 3512 4102 4103 4104 4105 4106 4107 4108 4200 4201 4202 4203 4204 4205 4206 4207)
partition=pi_mshoulde  #pg_tata  #mit_normal_gpu
reservation=orcd_testing # monthly_maint  # "" # orcd_testing  # WareWulf_testing 
gpu_type=l40s  # h200
gpu_count=4   # 8 # 4
subdir=output-l40s  # h200  # output-l40s
out_dir=/orcd/data/orcd/002/benchmarks/gpu-burn-r8/${partition}/${subdir}

mkdir -p $out_dir

for i in ${!nodes[@]}; do
        node=${nodes[i]}
	host=node$node
        sbatch << EOF
#!/bin/bash
#SBATCH -t 100
#SBATCH -p $partition
# #SBATCH -q unlimited
#SBATCH -N 1
#SBATCH -n $((2*$gpu_count))
#SBATCH --gres=gpu:$gpu_type:$gpu_count
#SBATCH -w $host
#SBATCH --mem=50GB
#SBATCH --reservation=$reservation

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

