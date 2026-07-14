#!/bin/bash
# Submit the 1-node NCCL test to Slurm (mit_testing partition), one job per node.
# Does the same work as run-nccl-1node.sh, but via Slurm instead of a manual ssh.
#
# Usage: ./job-nccl-1node.sh [nodes] [collectives] [ngpus]
#   nodes: comma/space separated node list (default: node5500 node5502).
#          One independent single-node job is submitted per node.
#   collectives: comma separated list or "all" (default: sendrecv). See
#                run-nccl-1node.sh for the full list of collective names.
#   ngpus: GPUs to use per node (default: auto-detect all GPUs).
#
# Examples:
#   ./job-nccl-1node.sh                              # both nodes, sendrecv, all GPUs
#   ./job-nccl-1node.sh node5500 allreduce           # node5500, allreduce, all GPUs
#   ./job-nccl-1node.sh node5500,node5502 all 8      # both nodes, all collectives, 8 GPUs
#
# The benchmark output lands in out-nccl-1node/ (written by run-nccl-1node.sh,
# same as a local run); Slurm's own stdout goes to slurm-logs/.

DIR=$(cd "$(dirname "$0")" && pwd)

NODES="${1:-node5500 node5502}"
COLLECTIVES="${2:-sendrecv}"
NGPUS="${3:-}"          # empty -> run-nccl-1node.sh auto-detects all GPUs

mkdir -p "$DIR/out-nccl-1node" "$DIR/slurm-logs"

for node in ${NODES//,/ }; do
   jid=$(sbatch --parsable \
      -p mit_testing -w "$node" -N 1 --exclusive \
      --gpus-per-node=b200:8 --mem=80GB -t 60 \
      -J "nccl-1node-$node" \
      -o "$DIR/slurm-logs/%x-%J.out" \
      --wrap "cd '$DIR' && ./run-nccl-1node.sh '$COLLECTIVES' '$NGPUS'")
   echo "Submitted nccl-1node on $node: job $jid"
done
