#nodes=(4505 4506 4507)
nodes=($1)
partition=$2  # mit_normal_gpu #ou_sloan_gpu  # mit_normal # mit_normal_gpu
reservation=$3 # orcd_testing # ""  #orcd_testing  #  WareWulf_testing
qos=$4 # unlimited 
cpu_count=$5 # 32  #48  # 176 # 96 # 88  # 120  # 240  # cores: reserved cores for GPUs

#echo $nodes
#echo $partition
#echo $reservation
#echo $qos
#echo $cpu_count

output_dir=/orcd/data/orcd/002/benchmarks/openmp/work/$partition/output

# don't change anything below
mkdir -p $output_dir
script_dir=/orcd/data/orcd/002/benchmarks/openmp/src/pi_omp
cores=$cpu_count
thread_list=(1)
threads=1
while [ $((threads * 4)) -lt $cores ]; do
        threads=$((2*threads))
        thread_list+=($threads)
done
thread_list+=($((cores / 2)) $cores $((3 * cores / 2)) $((2 * cores)))

# corner case handling - the formula above doesn't work well for < 4 cores (technically no node should have less than 4 cpus, computers these days have 12)
if [ $cores -lt 6 ]; then
        thread_list=(1 2 $cores)
fi

if [ $cores -eq 2 ]; then
        thread_list=(1 2)
fi

if [ $cores -eq 1 ]; then
        thread_list=(1)
fi

THREAD_LIST_STR="${thread_list[*]}"
echo "List of threads: $THREAD_LIST_STR"
export THREAD_LIST_STR

for i in ${!nodes[@]}; do
        host=node${nodes[i]}
        echo "running on host $host"
        sbatch << EOF
#!/bin/bash
#SBATCH -t 30
#SBATCH -p $partition
#SBATCH -n $cores
#SBATCH -N 1
#SBATCH -w $host
#SBATCH -o $output_dir/out_full.%N-%J
#SBATCH --reservation=$reservation
#SBATCH -q $qos
#SBATCH --exclusive

IFS=' ' read -a thread_list <<< "\$THREAD_LIST_STR"

echo "Benchmarking on \${thread_list[*]}"

for j in \${!thread_list[@]};
do
     export OMP_NUM_THREADS=\${thread_list[j]}
     echo "Ran with OMP_NUM_THREADS=\$OMP_NUM_THREADS"
     time $script_dir
done

EOF

done

