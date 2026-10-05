#!/usr/bin/env bash
# PART B (2-node) driver: smoke -> full collective sweep at PPN=8 -> PPN scaling -> analysis.
# Uses BOTH nodes, so nothing else may run on either. Run from node6100:
#   ./launch.sh node6100 rccl-tests/run_part_b_2node.sh
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
DRV=$LOG_ROOT/rccl/part_b_2node_$TS; mkdir -p "$DRV"
STATE=$DRV/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "PART B 2-node start (driver log: $DRV)"

# ---- GPU locks: node6100 first, then node6101 (fixed order -> no deadlock with other users) ----
# Other jobs on these nodes take /dev/shm/shaohao-gpu.lock per node. Hold both for the whole run:
# the local one on fd 9, the peer's through an ssh'd flock that lives until we close its stdin.
LOCK=/dev/shm/shaohao-gpu.lock
PEER=${PEER:-node6101.inband}
if [[ -z ${GPU_LOCKS_HELD:-} ]]; then
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
fi

# ---- stage 1: smoke (~2 min) -----------------------------------------------------
say "STAGE 1/4 smoke: all_reduce 16 ranks, 8B..1G"
rm -f "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"   # else a failed smoke reads the previous run's dir
COLLECTIVES=all_reduce MAX_BYTES=1G OUT_TAG=smoke ./run-rccl-2node.sh >"$DRV/smoke.log" 2>&1
rc=$?
smoke_dir=$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt" 2>/dev/null)
busbw=$(awk '$1=="all_reduce"{print $5}' "$smoke_dir/rccl_summary.txt" 2>/dev/null | tail -1)
say "smoke rc=$rc busbw@1G=${busbw:-none} GB/s dir=${smoke_dir:-none}"
if [[ $rc -ne 0 || -z "$busbw" || "$busbw" == "-" ]]; then
  say "ABORT: smoke produced no busbw. See $DRV/smoke.log and $smoke_dir/"
  [[ -n $smoke_dir ]] && tail -20 "$smoke_dir"/all_reduce_*.log 2>/dev/null | tee -a "$STATE"
  exit 1
fi
info=$(ls "$smoke_dir"/nccl_info_*.log 2>/dev/null | head -1)
if [[ -z $info ]] || grep -aq 'via NET/Socket' "$info" || ! grep -aq 'via NET/IB' "$info"; then
  say "ABORT: RCCL is not on NET/IB (RoCE ionic_*) -- see ${info:-$smoke_dir}"
  exit 1
fi
say "transport: $(grep -a -m1 -oE 'NET/IB : Using.{0,200}' "$info")"
[[ ${SMOKE_ONLY:-0} == 1 ]] && { say "SMOKE_ONLY=1 -> stop after smoke"; exit 0; }

# ---- stage 2: all collectives at PPN=8 (~30-60 min) --------------------------------
say "STAGE 2/4 collective sweep, PPN=8 (16 ranks)"
OUT_TAG=all ./run-rccl-2node.sh >"$DRV/all.log" 2>&1; rc=$?
all_dir=$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"); say "sweep rc=$rc -> $all_dir"

# ---- stage 3: per-node scaling, all_reduce + all_gather ----------------------------
say "STAGE 3/4 PPN scaling 1 2 4 8 (all_reduce, all_gather)"
PPN_LIST="1 2 4" COLLECTIVES="all_reduce all_gather" OUT_TAG=ppn ./run-rccl-2node.sh >"$DRV/ppn.log" 2>&1; rc=$?
ppn_dir=$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"); say "ppn sweep rc=$rc -> $ppn_dir"

# ---- stage 4: analysis ---------------------------------------------------------------
say "STAGE 4/4 analysis"
$PY analyze_rccl_2node.py "$all_dir" "$ppn_dir" -o "$RESULTS" >"$DRV/analyze.log" 2>&1
say "analyze rc=$? -> $RESULTS/rccl_2node.{md,csv}"
{ echo; echo "SMOKE_DIR=$smoke_dir"; echo "ALL_DIR=$all_dir"; echo "PPN_DIR=$ppn_dir"; } >>"$STATE"
say "PART B 2-node DONE"
