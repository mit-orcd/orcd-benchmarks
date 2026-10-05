#!/usr/bin/env bash
# Run the 2-node RCCL part on both ROCm stacks back to back (host 7.2.4, then 7.14) under ONE hold
# of the GPU locks, so no other job slips in between. Run from node6100:
#   ./launch.sh node6100 rccl-tests/run_part_b_2node_both.sh
# Lock order: node6100 first, then node6101 (same as run_part_b_2node.sh).
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks/amd-ubuntu
LOCK=/dev/shm/shaohao-gpu.lock
PEER=${PEER:-node6101.inband}
say() { echo "[$(date -Iseconds)] $*"; }
[[ $(hostname -s) == node6100 ]] || { say "run from node6100"; exit 1; }
say "waiting for $LOCK on node6100"
exec 9>>"$LOCK"; flock 9
say "got node6100 lock; waiting for $LOCK on $PEER"
coproc PEERLOCK { ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "$PEER" \
                  "flock $LOCK -c 'echo LOCKED; cat >/dev/null'"; }
read -r got <&"${PEERLOCK[0]}"
[[ $got == LOCKED ]] || { say "could not take $LOCK on $PEER"; exit 1; }
say "both nodes held"
export GPU_LOCKS_HELD=1
cd "$B/rccl-tests"
say "ROCm 7.2.4 (host)"; bash "$B/rccl-tests/run_part_b_2node.sh"; say "7.2.4 rc=$?"
say "ROCm 7.14";         bash "$B/with_rocm.sh" 7.14 rccl-tests/run_part_b_2node.sh; say "7.14 rc=$?"
say "done"
