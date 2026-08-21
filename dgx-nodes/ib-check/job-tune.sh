#!/bin/bash
#SBATCH -t 40
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -J nccl-tune
cd ${SLURM_SUBMIT_DIR:-.} || exit 1
source ./env-ubuntu.sh || exit 1
B=$BUILD_DIR/all_reduce_perf
ARGS="-b 1G -e 8G -f 2 -c 0"

echo "=== topology ==="
nvidia-smi topo -m 2>&1 | head -12
echo "=== cores ==="; nproc; echo "affinity: $(taskset -cp $$ 2>/dev/null)"

echo; echo "########## A: current -- np 1, g 8 (default binding) ##########"
mpirun -np 1 $MPI_FLAGS $B $ARGS -g 8 2>&1 | tail -6

echo; echo "########## B: np 1, g 8, --bind-to none ##########"
mpirun -np 1 --bind-to none $MPI_FLAGS $B $ARGS -g 8 2>&1 | tail -6

echo; echo "########## C: np 8, g 1, --bind-to numa (standard) ##########"
mpirun -np 8 --bind-to numa --map-by ppr:4:socket $MPI_FLAGS $B $ARGS -g 1 2>&1 | tail -6
