#!/bin/bash
# Submit gpu-fryer to Slurm (mit_testing partition), one job per selected node.
# Does the same work as run-gpu-fryer.sh, but via Slurm instead of a manual ssh.
#
# Usage: ./job-gpu-fryer.sh [nodes] [elapse_seconds]
#   nodes: comma/space separated node list (default: node5500 node5502).
#          One independent single-node job is submitted per node.
#   elapse_seconds: stress duration per precision (default: 300).
#
# Examples:
#   ./job-gpu-fryer.sh                    # both nodes, 300 s
#   ./job-gpu-fryer.sh node5500           # just node5500
#   ./job-gpu-fryer.sh node5500,node5502 600
#
# The benchmark output lands in out-gpu-fryer/ (written by run-gpu-fryer.sh,
# same as a local run); Slurm's own stdout goes to slurm-logs/.

DIR=$(cd "$(dirname "$0")" && pwd)

NODES="${1:-node5500 node5502}"
ELAPSE="${2:-300}"

mkdir -p "$DIR/out-gpu-fryer" "$DIR/slurm-logs"

for node in ${NODES//,/ }; do
   jid=$(sbatch --parsable \
      -p mit_testing -w "$node" -N 1 --exclusive \
      --gpus-per-node=b200:8 --mem=80GB -t 60 \
      -J "gpu-fryer-$node" \
      -o "$DIR/slurm-logs/%x-%J.out" \
      --wrap "cd '$DIR' && ./run-gpu-fryer.sh $ELAPSE")
   echo "Submitted gpu-fryer on $node: job $jid"
done
