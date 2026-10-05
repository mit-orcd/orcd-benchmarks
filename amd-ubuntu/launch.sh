#!/usr/bin/env bash
# Run a script on an AMD node, detached, so it survives logout (the nodes are ssh-only,
# not in Slurm). Run from a login node.
#
#   ./launch.sh node6100 work-rocmval/run_part_a.sh
#   ./launch.sh node6101 rccl-tests/run_part_b.sh
#   ./launch.sh node6100 rccl-tests/run-rccl-2node.sh      # uses both nodes
#
# Output: logs/launch/<node>_<script>_<ts>.log ; the script's own logs go to logs/<part>/.
set -euo pipefail
BENCH_ROOT=/orcd/data/orcd/022/benchmarks/amd-ubuntu
NODE=${1:?usage: $0 <node6100|node6101> <script relative to $BENCH_ROOT> [args...]}; shift
SCRIPT=${1:?script}; shift
[[ $NODE == *.inband ]] || NODE=$NODE.inband
[[ -f $BENCH_ROOT/$SCRIPT ]] || { echo "no such script: $BENCH_ROOT/$SCRIPT" >&2; exit 1; }
mkdir -p "$BENCH_ROOT/logs/launch"
LOG=$BENCH_ROOT/logs/launch/${NODE%%.*}_$(basename "$SCRIPT" .sh)_$(date +%Y%m%d_%H%M%S).log
ARGS=$(printf '%q ' "$@")
ssh -o BatchMode=yes "$NODE" \
  "cd $BENCH_ROOT/$(dirname "$SCRIPT") || exit 1; setsid nohup bash $BENCH_ROOT/$SCRIPT $ARGS > $LOG 2>&1 < /dev/null & echo pid \$!"
echo "log: $LOG"
