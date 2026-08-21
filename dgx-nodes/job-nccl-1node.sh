#!/bin/bash
#SBATCH -t 120
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=200GB
#SBATCH -J nccl-1node
#
# 1-node NCCL (intra-node, NVLink/NVSwitch) on one Ubuntu DGX H100 node.
# Submitted by ./run-nccl.sh, which supplies -p/-q/--gres/-w/-o.
#
# Usage (via run-nccl.sh):  sbatch <flags> job-nccl-1node.sh <gpus> [collectives]
#   gpus:        GPUs to use on the node (default 8)
#   collectives: comma separated short names, or "all" (default all)

cd ${SLURM_SUBMIT_DIR:-.} || exit 1
source ./env-ubuntu.sh || exit 1

GPUS_PER_TASK=${1:-8}
select_programs "${2:-all}" || exit 1

echo "node                = $(hostname)  ($(. /etc/os-release; echo "$PRETTY_NAME"))"
echo "driver / gpu        = $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1)"
echo "mpirun              = $(command -v mpirun)"
echo "Bin dir             = $BUILD_DIR"
echo "num_cpu = num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task    = $GPUS_PER_TASK"
echo "collectives         = ${PROGRAMS[*]}"

#export NCCL_DEBUG=INFO

for program in "${PROGRAMS[@]}"
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np 1 $MPI_FLAGS $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done

# "mpirun -np 1" runs one MPI task driving all the GPUs of the node; NCCL does
# the GPU-to-GPU communication over NVLink/NVSwitch.
