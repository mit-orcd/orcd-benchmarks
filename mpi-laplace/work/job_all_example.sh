#!/bin/bash

nodes=(3619)
partition=pi_keating
output_dir=/orcd/data/orcd/002/benchmarks/mpi-laplace/work/$partition/output

mkdir -p output_dir

for i in ${!nodes[@]}; do
	host=node${nodes[i]}
	sbatch << EOF
#!/bin/bash
#SBATCH -p $partition
#SBATCH -o $output_dir/%N-%J.out
#SBATCH -w $host
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=00:10:00
#SBATCH -J benchmark_mpi_laplace
#SBATCH --reservation=orcd_testing
##SBATCH --reservation=monthly_maint
##SBATCH -q unlimited

module load StdEnv
module load gcc
#module load openmpi/5.0.6
module load openmpi/4.1.4
#module load deprecated-modules
#module use /orcd/software/community/001/old_modulefiles/rocky8
#module load gcc/12.2.0-x86_64
#module load openmpi/4.1.4-pmi-ucx-x86_64

echo "========================="
srun hostname
which mpirun
mpirun hostname
lscpu
echo "========================="

BIN_DIR=/orcd/data/orcd/002/benchmarks/mpi-laplace/src/bin-r8 # For mpi 4.1.4?
#BIN_DIR=/orcd/data/orcd/002/benchmarks/mpi-laplace/src/bin-r8/mpi-5.0.6

\$BIN_DIR/laplace_serial < ./inp
echo "========================="
mpirun -np 4 \$BIN_DIR/laplace_mpi < ./inp
EOF
done
