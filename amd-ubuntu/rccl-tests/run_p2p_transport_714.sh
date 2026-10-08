#!/usr/bin/env bash
# 2-node alltoall + sendrecv on ROCm 7.14 (RCCL 2.30, new IB-CAST network transport) with
# different network settings: is the new transport why point-to-point is slower than on the
# host ROCm 7.2.4 (RCCL 2.27, plain IB transport)? Run from node6100 with the 7.14 stack:
#   ./launch.sh node6100 with_rocm.sh 7.14 rccl-tests/run_p2p_transport_714.sh
# Results: one line per run in $OUT/STATE.txt and $OUT/rccl_runs.txt.
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rccl/p2p_transport_714_$TS; mkdir -p "$OUT"
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
  PPN_LIST=$2 COLLECTIVES="alltoall sendrecv" MIN_BYTES=1M MAX_BYTES=16G OUT_TAG=p2p714_$1 EXTRA_ENV="$3" \
      ./run-rccl-2node.sh > "$OUT/rccl_$1.log" 2>&1
  local d; d=$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt" 2>/dev/null)
  echo "$1|$2|$3|$d" >> "$OUT/rccl_runs.txt"
  say "rccl $1 (PPN=$2, $3): $(awk '$1=="alltoall"||$1=="sendrecv"{printf "%s %s GB/s; ", $1, $5}' "$d/rccl_summary.txt" 2>/dev/null) transport: $(grep -ah -o -m1 'via NET/[A-Za-z-]*' "$d"/*.log 2>/dev/null | head -1)"
}
rccl default      8 "NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
rccl nch4         8 "NCCL_NCHANNELS_PER_NET_PEER=4 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
rccl netib        8 "NCCL_NET=IB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
rccl netib_nch4   8 "NCCL_NET=IB NCCL_NCHANNELS_PER_NET_PEER=4 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
rccl qpsched      8 "NCCL_IB_QP_SCHED_ENABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
rccl qpsched_nch4 8 "NCCL_IB_QP_SCHED_ENABLE=1 NCCL_NCHANNELS_PER_NET_PEER=4 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"

eval "exec ${PEERLOCK[1]}>&-"; flock -u 9
say "done, locks released"
