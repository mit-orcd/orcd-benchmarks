#!/bin/bash
#SBATCH -J mpi-io-bw
#SBATCH -p sched_mit_wvanrees  # sched_mit_psfc 
#SBATCH -t 06:00:00  # 6-00:00:00
#SBATCH -N 1
#SBATCH -n 32 
#SBATCH --mem=30GB   # 128GB
#SBATCH -o out/%x-%N-%j

module load gcc/6.2.0 openmpi/3.0.

DIR="/orcd/scratch/orcd/002/shaohao/mpi-io"
mkdir -p $DIR
EXE="mpi-io-bw-c7"
cp $EXE $DIR
cd $DIR
pwd
# mkdir out

for n in 1 2 4 8 16 32
do
  echo "====== Run with $n MPI tasks ======"
  mpirun -np $n ./mpi-io-bw
done

#rm -f $EXE

