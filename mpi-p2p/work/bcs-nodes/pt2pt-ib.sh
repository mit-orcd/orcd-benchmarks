#!/bin/bash
#SBATCH --reservation=mpi_test
#SBATCH -p ou_bcs_low
#SBATCH -t 10
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
# #SBATCH --cores-per-socket=1
# #SBATCH --ntasks-per-socket=1
#SBATCH --mem=10GB     # mem per node
# #SBATCH -w node[1802,1803] 
#SBATCH -w node[2802,2803] 

source ./env.sh $1 $2

OUT_DIR=out-openmpi-$2-N2802-2803
#OUT_DIR=out-nvhpc-$2-N2802-2803-v1
#OUT_DIR=out-nvhpc-$2-N1802-1803-v1

mkdir $OUT_DIR

echo "--- mpirun ---"
which mpirun
mpirun -n $SLURM_NTASKS hostname
echo "--- srun ---"
srun hostname

for i in `seq 1 4`; do
  echo "--- osu_bw $i ---"
  mpirun -npernode 1 set-ib$i.sh osu_bw >&$OUT_DIR/out.bw-$i&
done

wait


