#!/usr/bin/env bash
# Login-node waiter for mi355x-rocm10.1-vs-b200.md (the AMD nodes cannot push):
#   1. when node6101's extra 1-node runs are done: rebuild the md files on node6100, sync and push
#   2. when run_2node_fix_rocm10.1.sh is done (it rebuilds the md files itself): sync and push
# Checks every 5 min, up to 30 h.
#   setsid nohup bash rocm10.1/push_rocm10.1_b200.sh > logs/launch/push_rocm10.1_b200.log 2>&1 < /dev/null &
set -uo pipefail
A=/orcd/data/orcd/022/benchmarks/amd-ubuntu
B=/orcd/data/orcd/022/benchmarks
PY=$B/amd-software/venv/bin/python
CO="

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
say() { echo "[$(date -Iseconds)] $*"; }
wait_for() {  # $1 = command that succeeds when ready
  for i in $(seq 1 360); do eval "$1" && return 0; sleep 300; done; return 1
}
if wait_for "grep -q '1node extra DONE' $A/logs/launch/node6101_run_1node_extra_rocm10.1_*.log 2>/dev/null"; then
  say "node6101 runs done; rebuild md on node6100"
  ssh -o BatchMode=yes node6100.inband "cd $A && $PY rocm10.1/make_rocm10.1_vs_b200.py"
  $B/sync-and-push.sh "MI355X (ROCm 10.1) vs B200: new comparison file, 2-node results pending$CO"; say "push 1 rc=$?"
fi
if wait_for "test -f $A/logs/node6100/rocm10.1/driver/DONE2"; then
  say "2-node follow-up done: $(cat $A/logs/node6100/rocm10.1/driver/DONE2)"
  $B/sync-and-push.sh "MI355X (ROCm 10.1) vs B200: 2-node RCCL results (all_reduce data check, fixed setting, latency, bandwidth)$CO"; say "push 2 rc=$?"
fi
