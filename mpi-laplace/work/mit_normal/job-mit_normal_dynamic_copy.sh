#!/bin/bash

nodes=(1600 1601 1602 1603 1604 1605 1606 1607 1608 1609 1610 1611 1612 1613 1614 1616 1617 1618 1620 1621 1622 1623 1625)

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

