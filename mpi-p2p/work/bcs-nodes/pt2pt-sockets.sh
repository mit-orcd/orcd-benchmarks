#!/bin/bash
#SBATCH -p ou_bcs_low
#SBATCH -t 10
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#  #SBATCH --cores-per-socket=2
#SBATCH --ntasks-per-socket=4
#SBATCH --mem=10GB   # mem per node
#SBATCH --constraint=rocky8
#SBATCH -w node[1802,1803] 

source ./env.sh $1 $2

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

echo "--- osu_bw 1 ---"
mpirun --map-by node -n 2 osu_bw >&out.bw-1&
echo "--- osu_bw 2 ---"
mpirun --map-by node -n 2 osu_bw >&out.bw-2&
echo "--- osu_bw 3 ---"
mpirun --map-by node -n 2 osu_bw >&out.bw-3&
echo "--- osu_bw 4 ---"
mpirun --map-by node -n 2 osu_bw >&out.bw-4&
echo "--- osu_bw 5 ---"
mpirun --map-by node -n 2 osu_bw > out.bw-5
wait


