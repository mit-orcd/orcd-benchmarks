#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -t 30
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=8GB
#SBATCH --constraint=rocky8
#SBATCH -J osu-build
#SBATCH -o /home/shaohao/benchmarks/mpi-p2p/src/out.build-ompi-stacks-%j

# OSU builds for the MPI launcher test matrix (check-cluster mpi domain).
# openmpi/4.1.4 (spack, gcc 12.2.0) already has one: install/r8-4.1.4.
cd /home/shaohao/benchmarks/mpi-p2p/src

# (r8-spack-ompi-5.0.8 was built by the first run of this job, 2026-09-23)
# ./build-ompi-stack.sh r8-spack-ompi-5.0.8 \
#     /orcd/software/core/001/spack/modulefiles/gcc/12.2.0 openmpi/5.0.8
# core openmpi/5.0.8 links /opt/mellanox/hcoll, which only some nodes have
# (node1619 yes, node1363 no) -- hence -p mit_normal, whose nodes have it.
./build-ompi-stack.sh r8-core-ompi-5.0.8 \
    /orcd/software/core/001/modulefiles openmpi/5.0.8
