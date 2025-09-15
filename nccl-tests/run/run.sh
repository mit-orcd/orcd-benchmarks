
nodes=($1)
partition=$2 #mit_normal_gpu #pi_mghassem  # mit_normal 
reservation=$3 #orcd_testing  # "" #WareWulf_testing  # orcd_testing
qos=$4
cpu_count=$5   # 8 # 4
gpu_type=$6  #l40s  # h200
ngpus=$7

#echo $nodes
#echo $partition
#echo $reservation
#echo $qos
#echo $cpu_count
#echo $gpu_type
#echo $ngpus

out_dir=../${partition}/out-1node
mkdir -p $out_dir

flags="--reservation=$reservation -p $partition -q $qos --gres=gpu:$gpu_type:$ngpus --exclusive -o $out_dir/%x-%N-%J"

for i in ${!nodes[@]}
do
   host=node${nodes[i]}
   echo $host
   sbatch $flags -w $host 1node.sh $ngpus
done

