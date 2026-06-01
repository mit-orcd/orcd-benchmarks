#!/bin/bash
#nodes=(3206 3207 3208)
#nodes=(3301 3302 3402)
#nodes=(3402 3403)
#nodes=(3301 3403)
#nodes=(3301 3302 3402 3403 3405 3406 3407 3408 3500 3501 3502 3503 3504 3505 3507 3508 3509 3510 3511 3512)
#nodes=(4102 4103 4105 4106 4107 4108 4200 4201 4202 4203 4204 4205 4206)   # L40S
#nodes=(4209 4210 4211 4212 4302 4303 4304 4305 4502 4503 4504)   # l40s
#nodes=(3401 3001 4100)  # h200
#nodes=(4306 4307 4308)  # l40s
#nodes=(4505 4506 4507)
#nodes=(4300 4301)
nodes=(2300 2301)
gpu_type=h100 #h200  $l40s  # h200  # h200  # l40s

partition=sched_mit_psfc_gpu_r8  #mit_normal_gpu # pg_tata # pi_mshoulde #pg_tata #mit_normal_gpu

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
            sbatch $flags -w $host1,$host2  2nodes-2gpus.sh
        fi
    done
done

#sleep 50000
#mv out-2node-2gpu/* $output_dir

