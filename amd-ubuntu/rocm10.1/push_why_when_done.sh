#!/usr/bin/env bash
# Login-node waiter for run_rccl_why_1node.sh (the AMD nodes cannot push): waits for its DONE_why
# marker, then syncs and pushes. Checks every 2 min, up to 6 h.
#   setsid nohup bash rocm10.1/push_why_when_done.sh > logs/launch/push_why.log 2>&1 < /dev/null &
set -uo pipefail
DONE=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm10.1/driver/DONE_why
for i in $(seq 1 180); do
  [[ -f $DONE ]] && break
  sleep 120
done
[[ -f $DONE ]] || { echo "[$(date -Iseconds)] no DONE_why after 6 h, not pushing"; exit 1; }
echo "[$(date -Iseconds)] DONE_why: $(cat "$DONE"); sync and push"
/orcd/data/orcd/022/benchmarks/sync-and-push.sh "MI355X ROCm 10.1 vs 7.14: why 1-node RCCL latency changed (DDA path, debug + DDA-off runs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
echo "[$(date -Iseconds)] push rc=$?"
