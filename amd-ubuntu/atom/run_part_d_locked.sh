#!/usr/bin/env bash
# Wait until no foreign ATOM server is up on this node, then run run_part_d.sh under the
# node-local GPU lock (other agents run GPU jobs on these nodes concurrently).
#   setsid nohup atom/run_part_d_locked.sh "tier1 tier2" > log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
while pgrep -f '[a]tom.entrypoints' >/dev/null; do
  echo "$(date -Is) waiting: foreign ATOM server running"; sleep 300
done
echo "$(date -Is) acquiring /dev/shm/shaohao-gpu.lock"
exec flock /dev/shm/shaohao-gpu.lock bash ./run_part_d.sh "$@"
