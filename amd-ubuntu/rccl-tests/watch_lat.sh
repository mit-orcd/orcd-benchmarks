#!/usr/bin/env bash
# Background watcher for the latency runs: updates the md files as each part finishes.
#   MI355X 7.2.4 or 7.14 run DONE -> results/ubuntu/latency.md        (compare_lat.py ubuntu)
#   MI355X 7.14 DONE and B200 job done -> §2.3 of results/mi355x-vs-b200-interconnect.md (compare_lat.py b200)
# Run on the login node, detached:
#   setsid nohup bash rccl-tests/watch_lat.sh <b200 jobid> > logs/launch/watch_lat_<ts>.log 2>&1 &
# Gives up after MAX_H hours. Slurm controller outages are tolerated: the B200 job counts as
# finished only when its output says "done" or sacct reports a final state.
set -uo pipefail
BENCH=/orcd/data/orcd/022/benchmarks/amd-ubuntu
JOB=${1:?b200 jobid}
BOUT=/orcd/data/orcd/022/benchmarks/b200-nodes/out-nccl-lat
MAX_H=${MAX_H:-96}
PY=${PY:-python3}
say() { echo "[$(date -Iseconds)] $*"; }
done_amd() { ls "$BENCH"/logs/node6100/$1rccl/lat_2node_*/STATE.txt 2>/dev/null | xargs -r grep -l "latency run DONE" | grep -q .; }
b200_state() {
  if grep -q "^done " "$BOUT/nccl-lat-2node-$JOB.out" 2>/dev/null; then echo done; return; fi
  local s; s=$(sacct -n -X -j "$JOB" -o State 2>/dev/null | head -1 | tr -d ' +')
  case $s in COMPLETED|FAILED|TIMEOUT|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED) echo "ended:$s";; *) echo "${s:-unknown}";; esac
}

say "watching: MI355X 7.2.4 + 7.14 latency runs, B200 job $JOB"
u724=0; u714=0; ub=0; end=$(( $(date +%s) + MAX_H * 3600 ))
while (( $(date +%s) < end )); do
  if (( !u724 )) && done_amd ""; then u724=1; say "7.2.4 DONE"; (cd "$BENCH/rccl-tests" && $PY compare_lat.py ubuntu); fi
  if (( !u714 )) && done_amd "rocm7.14/"; then u714=1; say "7.14 DONE"; (cd "$BENCH/rccl-tests" && $PY compare_lat.py ubuntu); fi
  if (( !ub )); then
    st=$(b200_state)
    if [[ $st != done && $st == ended:* ]]; then
      ub=1; say "B200 job $JOB $st without finishing -- see $BOUT/nccl-lat-2node-$JOB.out"
      echo "$(date -Iseconds) latency: B200 job $JOB $st, interconnect §2.3 not written" >> "$BENCH/results/status.txt"
    elif [[ $st == done ]] && (( u714 )); then
      ub=1; say "B200 job $JOB done"
      (cd "$BENCH/rccl-tests" && $PY compare_lat.py b200 "$BOUT/$JOB") && \
        echo "$(date -Iseconds) latency: DONE, results/ubuntu/latency.md + mi355x-vs-b200-interconnect.md §2.3 updated (B200 job $JOB)" >> "$BENCH/results/status.txt"
    fi
  fi
  (( u724 && u714 && ub )) && { say "all done"; exit 0; }
  sleep 300
done
say "gave up after $MAX_H h: 7.2.4=$u724 7.14=$u714 b200=$ub"
echo "$(date -Iseconds) latency watcher gave up: 7.2.4=$u724 7.14=$u714 b200=$ub" >> "$BENCH/results/status.txt"
