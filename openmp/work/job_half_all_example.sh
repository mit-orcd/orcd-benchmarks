#!/bin/bash
#Author: justinwz
# just change the 4 variables in the header
nodes=(3404 3506)
partition=mit_normal_gpu
cpu_count=128
output_dir=/orcd/data/orcd/001/benchmarks/openmp/work/$partition/output


# don't change anything below
mkdir -p $output_dir
script_dir=/orcd/data/orcd/001/benchmarks/openmp/src/pi_omp
cores=$((cpu_count / 2))
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
#SBATCH -o $output_dir/out_half.%N-%J

IFS=' ' read -a thread_list <<< "\$THREAD_LIST_STR"

echo "Benchmarking on \${thread_list[*]}"
echo "CPU cores: $cores"
for j in \${!thread_list[@]}; 
do
     export OMP_NUM_THREADS=\${thread_list[j]}
     echo "Ran with OMP_NUM_THREADS=\$OMP_NUM_THREADS"
     time $script_dir
done

EOF

done

