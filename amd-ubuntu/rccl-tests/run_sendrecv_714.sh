#!/usr/bin/env bash
# 2-node sendrecv on ROCm 7.14: default vs NCCL_NCHANNELS_PER_NET_PEER=4/8 (the fix found
# on the host stack in run_sendrecv_check.sh). Run from node6100 with the 7.14 stack:
#   ./launch.sh node6100 with_rocm.sh 7.14 rccl-tests/run_sendrecv_714.sh
# Results: one line per run in $OUT/STATE.txt and $OUT/rccl_runs.txt.
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rccl/sendrecv_714_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "sendrecv check start ($OUT)"

# ---- GPU locks: node6100 first, then node6101 (same as run_part_b_2node.sh) ----
LOCK=/dev/shm/shaohao-gpu.lock
PEER=${PEER:-node6101.inband}
[[ $HOSTNAME_S == node6100 ]] || { say "ABORT: run from node6100 (lock order 6100 -> 6101)"; exit 1; }
say "waiting for $LOCK on $HOSTNAME_S"
exec 9>>"$LOCK"; flock 9
say "got $HOSTNAME_S lock; waiting for $LOCK on $PEER"
coproc PEERLOCK { ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "$PEER" \
                  "flock $LOCK -c 'echo LOCKED; cat >/dev/null'"; }
read -r got <&"${PEERLOCK[0]}"
[[ $got == LOCKED ]] || { say "ABORT: could not take $LOCK on $PEER"; exit 1; }
say "got $PEER lock -- both nodes held"
export GPU_LOCKS_HELD=1
# ---- B. RCCL sendrecv, one setting per run ----------------------------------------
rccl() {  # $1 tag  $2 PPN  $3 EXTRA_ENV
  rm -f "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"
  PPN_LIST=$2 COLLECTIVES=sendrecv MIN_BYTES=1M MAX_BYTES=16G OUT_TAG=sr714_$1 EXTRA_ENV="$3" \
      ./run-rccl-2node.sh > "$OUT/rccl_$1.log" 2>&1
  local d; d=$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt" 2>/dev/null)
  echo "$1|$2|$3|$d" >> "$OUT/rccl_runs.txt"
  say "rccl $1 (PPN=$2, $3): $(awk '$1=="sendrecv"{print $5" GB/s"}' "$d/rccl_summary.txt" 2>/dev/null)"
}
rccl default      8 ""
rccl nch4         8 "NCCL_NCHANNELS_PER_NET_PEER=4"
rccl nch8         8 "NCCL_NCHANNELS_PER_NET_PEER=8"

eval "exec ${PEERLOCK[1]}>&-"; flock -u 9
say "done, locks released"
