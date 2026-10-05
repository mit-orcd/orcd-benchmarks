#!/usr/bin/env bash
# Per-rank launcher for run-rccl-2node.sh: rccl-tests (MPI=1, -g 1) uses GPU = local rank, and
# RCCL pairs GPU r with its PCIe-local rail ionic_r. GPUs/rails 0-3 sit on NUMA0, 4-7 on NUMA1,
# so pin the rank's CPUs and host memory to that NUMA node.
r=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
n=$(( r < 4 ? 0 : 1 ))
exec numactl --cpunodebind=$n --membind=$n "$@"
