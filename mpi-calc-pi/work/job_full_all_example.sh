#!/bin/bash
#Author: justinwz
# just change the 4 variables in the header
#nodes=(4405 4406 4407 4408 4409 4410 4411 4412 4413)
#nodes=(4401 4402 4403 4404)
#nodes=(4208 4209 4210 4211 4212 4302 4303 4304 4305 4502 4503 4504)  # l40s
#nodes=(4104 4207)
#nodes=(3401 3001 4100)  # h200
#nodes=(3103 3104 3105 3106 3107 3108 3109 3110 3111 3112 3113 3114 3311 3312 3313 3314)
#nodes=(4306 4307 4308)  # l40s
#nodes=(4505 4506 4507)
nodes=(4505)
#nodes=(4100 3401)  # h200

partition=pi_mshoulde #pg_tata  #mit_normal_gpu  # ou_sloan_gpu  # mit_normal #mit_normal_gpu #pi_mghassem  # mit_normal  # CHANGE_ME: what partition are we using
reservation=orcd_testing  # "" #WareWulf_testing  # orcd_testing
cpu_count=32 #48  # 176  #96  #48  # 88  # 32  # 96  # 40  # 192             # CHANGE_ME: how many physical cores are on the node.
output_dir=/orcd/data/orcd/002/benchmarks/mpi-calc-pi/work/$partition/output   # CHANGE_ME: where the output results should be stored
echo "$output_dir"

# don't change anything below
mkdir -p $output_dir
script_dir=/orcd/data/orcd/002/benchmarks/mpi-calc-pi/src
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
# #SBATCH -q unlimited

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

