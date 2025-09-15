#!/bin/bash
nodes=($1)
partition=$2  # mit_normal_gpu #ou_sloan_gpu  # mit_normal # mit_normal_gpu
reservation=$3 # orcd_testing # ""  #orcd_testing  #  WareWulf_testing
qos=$4 # unlimited 

output_dir=/orcd/data/orcd/002/benchmarks/mpi-p2p/work/$partition/output
mkdir -p $output_dir
env_dir=/orcd/data/orcd/002/benchmarks/mpi-p2p/work

for i in ${!nodes[@]}; do
    for j in ${!nodes[@]}; do
        if [ $i -lt $j ]; then
            host1=node${nodes[i]}
            host2=node${nodes[j]}
            echo "Running on hosts ${host1} and ${host2}"
            sbatch << EOF
#!/bin/bash
#SBATCH -p $partition
#SBATCH -t 10
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10GB   # mem per node
#SBATCH --constraint=rocky8
#SBATCH -w ${host1},${host2}
#SBATCH -o $output_dir/out.${host1}_${host2}-%J
#SBATCH -J benchmark_mpi_p2p
#SBATCH --reservation=$reservation
#SBATCH --exclusive
#SBATCH -q $qos

echo "env_file: ${env_dir}/env.sh"

#source $env_dir/env.sh r8 4.1.4-pmi-ucx-x86_64
source $env_dir/env.sh r8 4.1.4-new

echo "number of nodes = \${SLURM_NNODES}"
echo "total number of tasks = \${SLURM_NTASKS}"
echo "number of tasks per core = \${SLURM_NTASKS_PER_CORE}"
echo "number of cores per task = \${SLURM_CPUS_PER_TASK}"
echo "total number of virtual cores (hyperthreads) per node = \${SLURM_CPUS_ON_NODE}"
echo "total memory per node = \${SLURM_MEM_PER_NODE}"

echo "--- mpirun ---"
which mpirun
echo "---srun---"
which srun
srun hostname
mpirun -n \${SLURM_NTASKS} hostname
echo "--- osu_bw ---"
mpirun -n \${SLURM_NTASKS} osu_bw
echo "--- osu_latency ---"
mpirun -n \${SLURM_NTASKS} osu_latency

EOF
        fi
	sleep 5
    done
done

