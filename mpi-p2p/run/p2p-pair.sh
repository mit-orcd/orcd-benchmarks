#!/bin/bash
# Quick inter-node MPI point-to-point health test between exactly two nodes,
# one rank per node. Used by check-cluster (mpi domain); also runnable by hand.
# run.sh (the full osu_bw + osu_latency sbatch campaign) is unchanged.
#
# p2p-pair.sh <nodeA> <nodeB> [partition] [build]
#   partition: default ou_orcd_everything (sched_system_all never hands out
#              allocations right now, even for idle nodes)
#   build:     OSU install tree under ../install (default r8-4.1.4, linked
#              against the openmpi/4.1.4 module -- NOT r8-5.0.8, which is linked
#              against nvhpc's openmpi-3.1.5)
#
# Launched with `srun --mpi=pmi2` directly, not salloc+mpirun: salloc can't get
# its allocation back to the login node ("Connection timed out"). One srun step
# can only run one MPI program, so it's a single osu_latency sweep from 1 B to
# 4 MiB: small-message latency from the 8 B row, and one-way bandwidth from
# the 4 MiB row (4194304 B / latency_us = MB/s).
#
# Last line of output (machine-readable):
#   P2P_RESULT lat_us=<8 B latency> big_lat_us=<4 MiB latency> bw_mbs=<MB/s> exec_s=<benchmark runtime>
# exec_s excludes Slurm queue wait. Exit code: 0 = benchmark ran, else failed.

nodeA=$1
nodeB=$2
partition=${3:-ou_orcd_everything}
build=${4:-r8-4.1.4}

if [[ -z "$nodeA" || -z "$nodeB" ]]; then
    echo "usage: $0 <nodeA> <nodeB> [partition] [build]" >&2
    exit 2
fi

type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
ompi_version=${build##*-}   # r8-4.1.4 -> 4.1.4
module load gcc/12.2.0 openmpi/"$ompi_version" 2>/dev/null
if ! command -v mpirun >/dev/null 2>&1; then
    echo "module load openmpi/$ompi_version FAILED" >&2
    echo "P2P_RESULT lat_us=NA big_lat_us=NA bw_mbs=NA exec_s=NA"
    exit 90
fi

INSTALL_ROOT=/orcd/data/orcd/022/benchmarks/mpi-p2p/install
OSU=${INSTALL_ROOT}/${build}/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency

# Short, non-exclusive 2-node step (1 core, a little memory per node) so
# busy-but-healthy nodes can still be tested. --immediate=120: give up instead of
# queueing forever; jobs here start on the backfill pass (bf_interval=30),
# which took ~50s even for idle nodes, so 30s was too short.
out=$(srun -p "$partition" -w "${nodeA},${nodeB}" -N 2 --ntasks-per-node=1 \
        --mem=2G -t 00:03:00 --immediate=120 -J mpi-p2p-check --mpi=pmi2 \
        bash -c 's=$(date +%s%N); "$0" -m 1:4194304 -i 200 -x 20; rc=$?;
                 echo "P2P_EXEC_NS=$(( $(date +%s%N) - s ))"; exit $rc' "$OSU" 2>&1)
rc=$?
echo "$out"

lat=$(echo "$out" | awk '$1==8 {print $2; exit}')
big=$(echo "$out" | awk '$1==4194304 {print $2; exit}')
ns=$(echo "$out" | sed -n 's/^P2P_EXEC_NS=//p' | sort -n | tail -1)
bw=NA
[ -n "$big" ] && bw=$(awk -v b="$big" 'BEGIN{ if (b>0) printf "%.0f", 4194304/b; else print "NA" }')
exec_s=NA
[ -n "$ns" ] && exec_s=$(awk -v n="$ns" 'BEGIN{ printf "%.2f", n/1e9 }')
echo "P2P_RESULT lat_us=${lat:-NA} big_lat_us=${big:-NA} bw_mbs=${bw} exec_s=${exec_s}"

[ $rc -eq 0 ] && [ -n "$lat" ] && [ -n "$big" ]
