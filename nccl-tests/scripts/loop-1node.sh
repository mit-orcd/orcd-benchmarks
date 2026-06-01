#!/bin/bash

#nodes=(4102 4103 4105 4106 4107 4108 4200 4201 4202 4203 4204 4205 4206)
#nodes=(2433 2434 3000 3100 3101 3300 3301 3400)  # h200
#nodes=(4209 4210 4211)
#nodes=(4212 4302 4303 4304 4305 4502 4503 4504)
#nodes=(3401 3001 4100)  # h200
#nodes=(4306 4307 4308)  # l40s
nodes=(4505 4506 4507)
#nodes=(4100 3401)  # h200

reservation=orcd_testing  #WareWulf_testing  #orcd_testing
partition=pi_mshoulde  # pg_tata # mit_normal_gpu
gpu_type=l40s #l40s  # h200
ngpus=4  # 4  #8 
qos=unlimited

flag="--reservation=$reservation -p $partition --gres=gpu:$gpu_type:$ngpus"
#flag="--reservation=$reservation -p $partition -q $qos --gres=gpu:$gpu_type:$ngpus"


for i in ${!nodes[@]}
do
   host=node${nodes[i]}
   echo $host
   sbatch $flag -w $host 1node.sh $ngpus
done

#sleep 600
#mv out-1node/* $output_dir
