#!/usr/bin/env bash
# Why does 2-node all_reduce latency differ between ROCm 7.14 and 10.1 (results/rocm10.1-vs-rocm7.14.md
# §5.2)? There, 7.14 runs RCCL's default but 10.1 runs NCCL_PROTO=LL128, because 10.1's default gives
# wrong results and 1.4-2.2 ms per call up to 256 KB (node6100/rocm10.1/fix_check.md). This run puts
# both stacks on the same settings, 2 nodes x 8 GPUs, all_reduce 8 B - 1 MB:
#   1. debug run (NCCL_DEBUG=INFO, TUNING,COLL): algorithm/protocol RCCL picks per size;
#   2. per setting: a check run (data check on, 20 iters) and a latency run (1000 iters + 100 warm-up,
#      no check, as in §5.2): default, NCCL_PROTO=LL128, NCCL_ALGO=Tree NCCL_PROTO=LL,
#      NCCL_ALGO=Ring NCCL_PROTO=LL; on 10.1 also RCCL_DDA_ENABLE=0.
# Then analyze_rccl_why_2node.py writes results/rccl-lat-2node-rocm10.1-why.md and §5.2.1 of
# results/rocm10.1-vs-rocm7.14.md; DONE_why2 tells push_why2_when_done.sh (login node) to push.
# Run from node6100 (lock order 6100 -> 6101):
#   ./launch.sh node6100 rocm10.1/run_rccl_why_2node.sh
set -uo pipefail
BR=/orcd/data/orcd/022/benchmarks/amd-ubuntu
TS=$(date +%Y%m%d_%H%M%S)
OUT=$BR/logs/node6100/rocm10.1/rccl/why_2node_$TS; mkdir -p "$OUT"
DONE=$BR/logs/node6100/rocm10.1/driver/DONE_why2; mkdir -p "$(dirname "$DONE")"; rm -f "$DONE"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "why-2node run start ($OUT)"
[[ $(hostname -s) == node6100 ]] || { say "ABORT: run from node6100"; echo "not node6100" > "$DONE"; exit 1; }

LOCK=/dev/shm/shaohao-gpu.lock
PEER=node6101.inband
say "waiting for $LOCK on node6100"
exec 9>>"$LOCK"; flock 9
say "got node6100 lock; waiting for $LOCK on $PEER"
coproc PEERLOCK { ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "$PEER" \
                  "flock $LOCK -c 'echo LOCKED; cat >/dev/null'"; }
read -r got <&"${PEERLOCK[0]}"
[[ $got == LOCKED ]] || { say "ABORT: could not take $LOCK on $PEER"; echo "no peer lock" > "$DONE"; exit 1; }
say "got $PEER lock -- both nodes held"
export GPU_LOCKS_HELD=1

# $1 stack  $2 tag  $3 iters  $4 check  $5 EXTRA_ENV  [NCCL_DEBUG/NCCL_DEBUG_SUBSYS from the caller]
run() {
  ( export ROCM_STACK=$1; unset RESULTS_SUBDIR
    source $BR/common/env.sh; cd $BR/rccl-tests
    rm -f "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"
    HOSTS="node6100.inband node6101.inband" PPN_LIST=8 COLLECTIVES=all_reduce MIN_BYTES=8 MAX_BYTES=1M \
      ITERS=$3 WARMUP=$(( $3 >= 1000 ? 100 : 5 )) CHECK=$4 OUT_TAG=why2n_$2 EXTRA_ENV="$5" \
      ./run-rccl-2node.sh > "$OUT/rccl_$1_$2.log" 2>&1
    echo "$1|$2|$5|$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt" 2>/dev/null)" >> "$OUT/runs.txt" )
  say "$1 $2 (${5:-default}): $(tail -1 "$OUT/runs.txt" | cut -d'|' -f4)"
}

SETTINGS=("default|" "ll128|NCCL_PROTO=LL128" "treell|NCCL_ALGO=Tree NCCL_PROTO=LL" "ringll|NCCL_ALGO=Ring NCCL_PROTO=LL")
for s in 7.14 10.1; do
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL run $s debug 2 1 ""
  v=("${SETTINGS[@]}"); [[ $s == 10.1 ]] && v+=("nodda|RCCL_DDA_ENABLE=0")
  for x in "${v[@]}"; do
    run $s "${x%%|*}_check" 20 1 "${x#*|}"
    run $s "${x%%|*}_lat" 1000 0 "${x#*|}"
  done
done
exec {PEERLOCK[1]}>&-
exec 9>&-
say "tests done, locks released"

( export ROCM_STACK=10.1; source $BR/common/env.sh
  $PY $BR/rocm10.1/analyze_rccl_why_2node.py "$OUT" ) > "$OUT/analyze.log" 2>&1
say "analyze rc=$?"
echo "$OUT" > "$DONE"
say "DONE"
