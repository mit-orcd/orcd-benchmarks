#!/usr/bin/env bash
# Login-node waiter: the AMD nodes cannot push (no git login, no ssh to the login nodes), so this
# waits for run_rocm10.1_all.sh's DONE marker, then syncs and pushes. Checks every 5 min, up to 30 h.
#   setsid nohup bash rocm10.1/push_when_done.sh > logs/launch/push_rocm10.1.log 2>&1 < /dev/null &
set -uo pipefail
DONE=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/rocm10.1/driver/DONE
for i in $(seq 1 360); do
  [[ -f $DONE ]] && break
  sleep 300
done
[[ -f $DONE ]] || { echo "[$(date -Iseconds)] no DONE after 30 h, not pushing"; exit 1; }
echo "[$(date -Iseconds)] DONE: $(cat "$DONE"); sync and push"
/orcd/data/orcd/022/benchmarks/sync-and-push.sh "ROCm 10.1 (hipBLASLt 1.4.1, RCCL 2.30.7) vs ROCm 7.14 on MI355X: install, scripts, results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
echo "[$(date -Iseconds)] push rc=$?"
