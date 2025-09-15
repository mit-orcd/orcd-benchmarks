#!/bin/bash
#SBATCH -J mpi-io-bw
#SBATCH -p mit_normal
#SBATCH -q unlimited
#SBATCH -t 06:00:00  # 6-00:00:00
#SBATCH -N 1
#SBATCH -n 32  # 64   # 96 
#SBATCH --mem=30GB   # 128GB
#SBATCH -o out/%x-%N-%j

module load openmpi/5.0.6

DIR="/orcd/scratch/orcd/002/shaohao/mpi-io"
mkdir -p $DIR
EXE="mpi-io-bw"
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

