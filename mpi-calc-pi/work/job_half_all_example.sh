#!/bin/bash
#Author: justinwz
# just change the 4 variables in the header
nodes=(1622 1623 1624 1625)                                                              # CHANGE ME: what nodes will we run the job on?
partition=mit_normal                                                     # CHANGE_ME: what partition are we using
cpu_count=192                                                              # CHANGE_ME: how many physical cores are on the node. We will request half of them
output_dir=/orcd/data/orcd/001/benchmarks/mpi-calc-pi/work/$partition/output   # CHANGE_ME: where the output results should be stored
echo "$output_dir"





# don't change anything below
mkdir -p $output_dir
script_dir=/orcd/data/orcd/001/benchmarks/mpi-calc-pi/src
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
#SBATCH --mem=0
#SBATCH -o $output_dir/out_half.%N-%J

IFS=' ' read -a thread_list <<< "\$THREAD_LIST_STR"

echo "Benchmarking on \${thread_list[*]}"

module load deprecated-modules
module use /orcd/software/community/001/old_modulefiles/rocky8
module load gcc/12.2.0-x86_64
module load openmpi/4.1.4-pmi-ucx-x86_64

which mpirun

for j in \${!thread_list[@]}; 
do
     export NUM_THREADS=\${thread_list[j]}
     echo "Ran with MPI_NUM_THREADS=\$NUM_THREADS"
     mpirun --oversubscribe -np \${NUM_THREADS} ${script_dir}/calc_pi_mpi # oversubscribe keyword allows hyperthreading
done

EOF

done

