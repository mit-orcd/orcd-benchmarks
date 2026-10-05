#!/usr/bin/env bash
# Wait until no foreign ATOM server is up on this node, then run run_part_d.sh under the
# node-local GPU lock (other agents run GPU jobs on these nodes concurrently). Once the lock
# is held, give the previous holder's GPU processes up to 10 min to drain before
# run_part_d.sh's own "GPUs busy" guard runs (it aborts immediately otherwise).
#   setsid nohup atom/run_part_d_locked.sh "tier1 tier2" > log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" != --locked ]]; then
  while pgrep -f '[a]tom.entrypoints' >/dev/null; do
    echo "$(date -Is) waiting: foreign ATOM server running"; sleep 300
  done
  echo "$(date -Is) acquiring /dev/shm/shaohao-gpu.lock"
  exec flock /dev/shm/shaohao-gpu.lock bash "$0" --locked "$@"
fi
shift
echo "$(date -Is) lock held"
for i in $(seq 1 60); do
  busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
  [[ "${busy:-0}" -eq 0 ]] && break
  echo "$(date -Is) $busy GPU(s) still busy, waiting"; sleep 10
done
exec bash ./run_part_d.sh "$@"
