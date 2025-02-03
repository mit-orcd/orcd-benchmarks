#!/bin/bash
# Run:
# mpirun -np $np $options set-ib.sh $exec

# local MPI rank on a node
export index=$OMPI_COMM_WORLD_LOCAL_RANK

if (( $index == 0 )); then
    export UCX_NET_DEVICES=mlx5_0:1
elif (( $index == 1 )); then
    export UCX_NET_DEVICES=mlx5_1:1
elif (( $index == 2 )); then
    export UCX_NET_DEVICES=mlx5_2:1
elif (( $index == 3 )); then
    export UCX_NET_DEVICES=mlx5_3:1
fi

echo "`hostname`: MPI local rank $index using hca $UCX_NET_DEVICES"
exec $*


