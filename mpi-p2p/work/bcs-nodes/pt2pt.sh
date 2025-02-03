#!/bin/bash
#SBATCH -p ou_bcs_low
#SBATCH -t 10
#SBATCH -N 2
#  #SBATCH -n 2    # not recommended, could be not evenly on each node
#SBATCH --ntasks-per-node=1
#  #SBATCH -c 4 
#  #SBATCH --ntasks-per-core=1          # Turn off hyperthreads
#SBATCH --mem=10GB   # mem per node
#SBATCH --constraint=rocky8
#SBATCH -w node[2702,2703]  # node[1802,1803] 

source ./env.sh $1 $2

#
echo "number of nodes = $SLURM_NNODES"

# -n --> SLURM_NTASKS
echo "total number of tasks = $SLURM_NTASKS"

# Turn off hyperthreads
echo "number of tasks per core = $SLURM_NTASKS_PER_CORE"

# ntasks-per-node --> SLURM_NTASKS_PER_NODE

# -c --> SLURM_CPUS_PER_TASK
echo "number of cores per task = $SLURM_CPUS_PER_TASK"

#
echo "total number of virutal cores (hyperthreads) per node = $SLURM_CPUS_ON_NODE"
echo "total memory per node = $SLURM_MEM_PER_NODE"


echo "--- mpirun ---"
which mpirun
mpirun -n $SLURM_NTASKS hostname
echo "--- srun ---"
srun hostname
echo "--- osu_bw ---"
mpirun -n $SLURM_NTASKS osu_bw
#mpirun --mca pml ucx -n $SLURM_NTASKS osu_bw
echo "--- osu_latency ---"
mpirun -n $SLURM_NTASKS osu_latency
#mpirun --mca pml ucx -n $SLURM_NTASKS osu_latency



