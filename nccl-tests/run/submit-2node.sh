nodes=(4300 4301)
#nodes=(3200 3201)
partition=pg_tata  # mit_normal_gpu #pi_mghassem  # mit_normal 
reservation=orcd_testing  # "" #WareWulf_testing  # orcd_testing
qos=unlimited
gpu_type=h200  #l40s  # h200

echo $nodes
echo $partition
echo $reservation
echo $qos
echo $gpu_type

out_dir=../${partition}/out-2node
mkdir -p $out_dir

flags="--reservation=$reservation -p $partition -q $qos --gpus-per-node=$gpu_type:1 --exclusive -o $out_dir/%x-%N-%J"
#flags="-p $partition -q $qos --gpus-per-node=$gpu_type:1 --exclusive -o $out_dir/%x-%N-%J"

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


