#!/bin/bash

nodes=(2433 2434 3000 3100 3101 3300 3301 3400)  # h200

partition=mit_normal_gpu
ngpus=4  #4, 8
#output_dir="../$partition/out-1node"
output_dir="../$partition/out-1node-maint"
#flag="--reservation=orcd_testing -p $partition -q unlimited --gres=gpu:$ngpus -o $output_dir"
#flag="--reservation=orcd_testing -p $partition -q unlimited --gres=gpu:$ngpus"
flag="--reservation=monthly_maint -p $partition -q unlimited --gres=gpu:$ngpus"

mkdir -p $output_dir

for i in ${!nodes[@]}
do
   host=node${nodes[i]}
   echo $host
   sbatch $flag -w $host 1node.sh $ngpus
done

sleep 600
mv out-1node/* $output_dir
