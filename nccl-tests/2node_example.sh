#!/bin/bash
# runs the NCCL benchmarks on a specified partition

# change these parameters
nodes=(1632 1633)
partition=pi_ashia07
out_dir=/orcd/data/orcd/001/benchmarks/nccl_tests/${partition}/2nodes_output

#DON'T change these parameters
# tasks_per_node=1 # don't think this matters? b/c we always put 1 task in mpirun
JOB_NAME=nvhpc-23.3-ompi3
BUILD_DIR=/orcd/data/orcd/001/benchmarks/nccl-tests/build-${JOB_NAME}

mkdir -p ${out_dir}

for i in ${!nodes[@]}; do 
       for j in ${!nodes[@]}; do
	       if [ $i -lt $j ]; then
		       host1=node${nodes[$i]}
		       host2=node${nodes[$j]}
		       echo "Running on $host1 and $host2"
		       sbatch << EOF
#!/bin/bash
#SBATCH -p $partition
#SBATCH -t 30
#SBATCH -N 2
#SBATCH -w $host1,$host2
#SBATCH -n 2
#SBATCH --gpus-per-node=1
#SBATCH -o $out_dir/out.%N-%J

module purge

module use /software/modulefiles
module load nvhpc/2023_233/nvhpc/23.3
#module load nvhpc/2024_245/24.5
#module load nvhpc/23.3

mpirun hostname
which mpirun
which nvcc
echo "Bin dir = $BUILD_DIR"

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4

echo "num_cpu = num_mpi_tasks = \$SLURM_NTASKS" # should be equal to tasks_per_node
echo "num_gpu_per_task = \$SLURM_GPUS_PER_NODE" # should be equal to 1

#export NCCL_DEBUG=INFO

for program in sendrecv_perf reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% ${BUILD_DIR}/\${program} %%%%%%%%%%"
   mpirun -np 2 --mca btl_openib_warn_no_device_params_found 0 ${BUILD_DIR}/\${program} -b \$MIN_SIZE -e \$MAX_SIZE -f \$FACTOR -g \$SLURM_GPUS_PER_NODE
done

EOF
fi
done
done

