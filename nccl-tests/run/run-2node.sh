#!/bin/bash
nodes=($1)
partition=$2 #mit_normal_gpu #pi_mghassem  # mit_normal 
reservation=$3 #orcd_testing  # "" #WareWulf_testing  # orcd_testing
qos=$4
cpu_count=$5   # 8 # 4
gpu_type=$6  #l40s  # h200
ngpus=$7

out_dir=../${partition}/out-2node
mkdir -p $out_dir

flags="--reservation=$reservation -p $partition -q $qos --gpus-per-node=$gpu_type:1 --exclusive -o $out_dir/%x-%N-%J"

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


