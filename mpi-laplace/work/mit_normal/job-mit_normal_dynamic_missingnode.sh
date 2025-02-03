#!/bin/bash

nodes=(1615 1619 1624)

for node in "${nodes[@]}"; do
    sbatch << EOF
#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -o out_files_copy/out.%N-%J
#SBATCH -w node$node
#SBATCH -N 1
#SBATCH -n 8

module use /orcd/software/community/001/modulefiles
module load gcc/12.2.0-x86_64
module load openmpi/4.1.4-pmi-ucx-x86_64.lua


echo "========================="
hostname
which mpirun
lscpu
echo "========================="

BIN_DIR=/orcd/data/orcd/001/benchmarks/mpi-laplace/bin-r8

\$BIN_DIR/laplace_serial < ../inp
echo "========================="
mpirun -np 4 \$BIN_DIR/laplace_mpi < ../inp

EOF

done

