#!/usr/bin/env bash
# ROCm 10.1 (RCCL 2.30.7), node6100 + node6101: 2-node RCCL follow-up.
#   A. all_reduce, 2 nodes x 8 GPUs, 8 B-4 MB, data check on (-c 1), RCCL default and fixed settings:
#      the default gives wrong results at 32-256 KB and takes 1.4-2.2 ms up to 256 KB. One debug run
#      logs RCCL's algorithm choice (NCCL_DEBUG_SUBSYS=TUNING).
#   B. pick_fix_rocm10.1.py -> results/node6100/rocm10.1/fix_choice.json + fix_check.md
#   C. 2-node latency again (run_rccl_lat_2node_rocm10.1.sh), plus the chosen fix as a third setting
#   D. 2-node bandwidth extras, as for the other stack: alltoall + sendrecv with NCCL_NET=IB, sendrecv with
#      4 channels per network peer (with and without NCCL_NET=IB), all_reduce with the fix (data check on)
#   E. make_rocm10.1_vs_b200.py (md files); writes logs/node6100/rocm10.1/driver/DONE2
# Run from node6100 (lock order 6100 -> 6101):
#   ./launch.sh node6100 rocm10.1/run_2node_fix_rocm10.1.sh
set -uo pipefail
export ROCM_STACK=10.1
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
HERE=$BENCH_ROOT/rocm10.1
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rccl/fix_2node_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
DRV=$LOG_ROOT/driver; mkdir -p "$DRV"; rm -f "$DRV/DONE2"
echo "$OUT" > "$RESULTS/fix_2node_dir.txt"
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "2-node follow-up start ($OUT)"
[[ $HOSTNAME_S == node6100 ]] || { say "ABORT: run from node6100"; exit 1; }

LOCK=/dev/shm/shaohao-gpu.lock
PEER=${PEER:-node6101.inband}
say "waiting for $LOCK on $HOSTNAME_S"
exec 9>>"$LOCK"; flock 9
say "got $HOSTNAME_S lock; waiting for $LOCK on $PEER"
coproc PEERLOCK { ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "$PEER" \
                  "flock $LOCK -c 'echo LOCKED; cat >/dev/null'"; }
read -r got <&"${PEERLOCK[0]}"
[[ $got == LOCKED ]] || { say "ABORT: could not take $LOCK on $PEER"; echo "no peer lock" > "$DRV/DONE2"; exit 1; }
say "got $PEER lock -- both nodes held"
export GPU_LOCKS_HELD=1
cd "$BENCH_ROOT/rccl-tests"
TWO="node6100.inband node6101.inband"

run() {  # $1 tag  $2 collectives  $3 min  $4 max  $5 iters  $6 check  $7 EXTRA_ENV  [$8 runs file]
  rm -f "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"
  HOSTS="$TWO" PPN_LIST=8 COLLECTIVES="$2" MIN_BYTES=$3 MAX_BYTES=$4 ALLTOALL_MAX=4G \
    ITERS=$5 WARMUP=10 CHECK=$6 OUT_TAG=fix_$1 EXTRA_ENV="$7" ./run-rccl-2node.sh > "$OUT/rccl_$1.log" 2>&1
  local d; d=$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt" 2>/dev/null)
  echo "$1|$7|$d" >> "$OUT/${8:-check_runs.txt}"
  say "$1 (${7:-default}): $d"
}

# ---- A. all_reduce check runs -------------------------------------------------------------
VARIANTS=("default|" "netib|NCCL_NET=IB" "ring|NCCL_ALGO=Ring" "tree|NCCL_ALGO=Tree"
          "simple|NCCL_PROTO=Simple" "ll|NCCL_PROTO=LL" "ll128|NCCL_PROTO=LL128"
          "nomscclpp|RCCL_MSCCLPP_ENABLE=0" "nomsccl|RCCL_MSCCL_ENABLE=0" "ringll|NCCL_ALGO=Ring NCCL_PROTO=LL")
for v in "${VARIANTS[@]}"; do run "${v%%|*}" all_reduce 8 4M 100 1 "${v#*|}"; sleep 5; done
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL run debug all_reduce 8 256K 2 1 "" debug_runs.txt

# ---- B. pick the fix ------------------------------------------------------------------------
$PY "$HERE/pick_fix_rocm10.1.py" "$OUT" 2>&1 | tee -a "$STATE"
FIX_ENV=$($PY -c "import json;print(json.load(open('$RESULTS/fix_choice.json')).get('fix_env',''))" 2>/dev/null)
say "fix: '${FIX_ENV:-none}'"

# ---- C. 2-node latency again (+ fix) ---------------------------------------------------------
LAT_EXTRA_SETTINGS=${FIX_ENV:+"fix|$FIX_ENV"} bash "$HERE/run_rccl_lat_2node_rocm10.1.sh" > "$OUT/lat_2node.log" 2>&1
say "C. latency rc=$? -> $RESULTS/rccl_lat.md"

# ---- D. bandwidth extras ------------------------------------------------------------------------
run bw_netib     "alltoall sendrecv" 8 16G 20 1 "NCCL_NET=IB"                                   bw_runs.txt
run bw_netib_ch4 sendrecv            8 16G 20 1 "NCCL_NET=IB NCCL_NCHANNELS_PER_NET_PEER=4"     bw_runs.txt
run bw_ch4       sendrecv            8 16G 20 1 "NCCL_NCHANNELS_PER_NET_PEER=4"                 bw_runs.txt
[[ -n $FIX_ENV ]] && run bw_fix "all_reduce" 8 16G 20 1 "$FIX_ENV" bw_runs.txt
exec {PEERLOCK[1]}>&-
exec 9>&-
say "tests done, locks released"

# ---- E. md files ----------------------------------------------------------------------------------
$PY "$HERE/make_rocm10.1_vs_b200.py" >> "$STATE" 2>&1; say "E. md rc=$?"
date -Iseconds > "$DRV/DONE2"
say "2-node follow-up DONE"
