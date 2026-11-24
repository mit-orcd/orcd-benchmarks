#!/bin/bash
#SBATCH -J cuda-mpi-b200
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=b200:8
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o %x-%j.out

# CUDA-aware MPI across B200 nodes with the HPC-X Open MPI in nvhpc/26.1.
# One MPI rank per GPU. Set APP to your program, e.g.
#   sbatch --export=ALL,APP="./my_app arg1" job-cuda-mpi-b200.sh
APP=${APP:-./my_app}

module purge
module load nvhpc/26.1
OMPI=$NVHPC_ROOT/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi
export PATH=$OMPI/bin:$PATH LD_LIBRARY_PATH=$OMPI/lib:$LD_LIBRARY_PATH

# Use only the 8 NDR cards. mlx5_0-3 are node-local (NVSwitch management)
# and make MPI fail with "UD endpoint ... unhandled timeout error".
export UCX_NET_DEVICES=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1

# local rank i -> i-th GPU of the job; --bind-to none avoids binding outside the Slurm CPUs
mpirun -np $SLURM_NTASKS --map-by ppr:$SLURM_NTASKS_PER_NODE:node --bind-to none \
    -x PATH -x LD_LIBRARY_PATH -x UCX_NET_DEVICES \
    bash -c 'g=(${CUDA_VISIBLE_DEVICES//,/ }); export CUDA_VISIBLE_DEVICES=${g[$OMPI_COMM_WORLD_LOCAL_RANK]}; exec '"$APP"
