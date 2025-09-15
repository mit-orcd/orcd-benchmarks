#!/bin/bash
# Run:
# mpirun -np $np $options set-ib.sh $exec

if (( $OMPI_COMM_WORLD_RANK == 0 )); then
  export UCX_NET_DEVICES=mlx5_1:1
elif (( $OMPI_COMM_WORLD_RANK == 1 )); then
  export UCX_NET_DEVICES=mlx5_1:1
fi

echo "`hostname`: MPI rank $OMPI_COMM_WORLD_RANK using hca $UCX_NET_DEVICES"
exec $*


