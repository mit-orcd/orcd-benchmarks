#!/bin/bash
#nodes=(1917 1918 2100 2101 2102 2103 2104 2119 2300 2301 2302 2303 2304 2319)
nodes=(9902 9903 9904 9905)
gpu_type=a100 #h200  $l40s  # h200  # h200  # l40s
partition=pi_mbathe  #sched_mit_psfc_gpu_r8

#output_dir="../$partition/out-2node-2gpu"
#mkdir -p $output_dir

#flags="--reservation=orcd_testing -p $partition -q unlimited --gpus-per-node=$gpu_type:1"
#flags="--reservation=monthly_maint -p $partition -q unlimited --gpus-per-node=$gpu_type:1 -o test3/%x-%N-%J"
#flags="--reservation=orcd_testing -p $partition -q unlimited --gpus-per-node=$gpu_type:1 -o test3/%x-%N-%J"
#flags="--reservation=orcd_testing -p $partition --gpus-per-node=$gpu_type:1 -o out-2node-2gpu/%x-%N-%J"
flags="-p $partition --gpus-per-node=$gpu_type:1 -o out-2node-2gpu/%x-%N-%J"


for i in ${!nodes[@]}; do
    for j in ${!nodes[@]}; do
        if [ $i -lt $j ]; then
            host1=node${nodes[i]}
            host2=node${nodes[j]}
            echo "Running on hosts ${host1} and ${host2}"
            sbatch $flags -w $host1,$host2  2nodes-avx2.sh 
        fi
    done
done

sleep 1
#mv out-2node-2gpu/* $output_dir

