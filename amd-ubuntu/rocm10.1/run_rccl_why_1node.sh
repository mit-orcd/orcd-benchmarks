#!/usr/bin/env bash
# Why did 1-node RCCL latency change from ROCm 7.14 to 10.1 (results/rocm10.1-vs-rocm7.14.md §5.1)?
# On node6100, under one GPU lock:
#   1. debug runs, both stacks: NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL,ENV into per-rank
#      files (NCCL_DEBUG_FILE, so stdout stays clean), sizes 8 B - 1 MB, 5 iters, one collective at a
#      time -> which implementation (DDA, MSCCL, ring/tree) and settings RCCL picks;
#   2. timing, default, both stacks: same as run_rccl_lat_1node.sh (1000 iters + 100 warm-up);
#   3. timing, 10.1 with RCCL_DDA_ENABLE=0: does the gap to 7.14 go away without the new DDA path?
# Then analyze_rccl_why_1node.py writes results/rccl-lat-1node-rocm10.1-why.md and §5.1.1 of
# results/rocm10.1-vs-rocm7.14.md, and DONE_why tells push_why_when_done.sh (login node) to push.
#   ./launch.sh node6100 rocm10.1/run_rccl_why_1node.sh
set -uo pipefail
BR=/orcd/data/orcd/022/benchmarks/amd-ubuntu
TS=$(date +%Y%m%d_%H%M%S)
OUT=$BR/logs/$(hostname -s)/rocm10.1/rccl/why_1node_$TS; mkdir -p "$OUT"
DONE=$BR/logs/$(hostname -s)/rocm10.1/driver/DONE_why; mkdir -p "$(dirname "$DONE")"; rm -f "$DONE"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "why-1node run start ($OUT)"

LOCK=/dev/shm/shaohao-gpu.lock
say "waiting for $LOCK"
exec 9>>"$LOCK"; flock 9
say "got lock"
export GPU_LOCKS_HELD=1

DBG_COLLS="sendrecv all_reduce all_gather broadcast alltoall"
TIME_COLLS="sendrecv all_reduce all_gather reduce_scatter broadcast alltoall"

# $1 stack  $2 tag  $3 EXTRA_ENV: timing run, like run_rccl_lat_1node.sh
timing() {
  ( export ROCM_STACK=$1; unset RESULTS_SUBDIR
    source $BR/common/env.sh; cd $BR/rccl-tests
    rm -f "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"
    HOSTS="$HOSTNAME_S.inband" PPN_LIST=8 COLLECTIVES="$TIME_COLLS" MIN_BYTES=8 MAX_BYTES=4M ALLTOALL_MAX=4M \
      ITERS=1000 WARMUP=100 CHECK=0 OUT_TAG=why1n_$2 EXTRA_ENV="$3" \
      ./run-rccl-2node.sh > "$OUT/time_$1_$2.log" 2>&1
    cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt" > "$OUT/time_$1_$2.dir" )
  say "timing $1 $2 (${3:-default}) -> $(cat "$OUT/time_$1_$2.dir" 2>/dev/null)"
}

# $1 stack: debug runs, one collective per mpirun, per-rank debug files
debug() {
  ( export ROCM_STACK=$1; unset RESULTS_SUBDIR
    source $BR/common/env.sh; cd $BR/rccl-tests
    for coll in $DBG_COLLS; do
      d=$OUT/debug_$1/$coll; mkdir -p "$d"
      timeout 600 "$MPI_HOME/bin/mpirun" --prefix "$MPI_HOME" -np 8 --host "$HOSTNAME_S.inband:8" \
          --map-by ppr:8:node --bind-to none --mca plm_rsh_args "-o BatchMode=yes" \
          --mca btl tcp,self --mca btl_tcp_if_include eno17695np0 --mca oob_tcp_if_include eno17695np0 \
          -x PATH -x LD_LIBRARY_PATH -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL,ENV \
          -x NCCL_DEBUG_FILE="$d/rank.%h.%p.log" \
          $BR/rccl-tests/numa-bind.sh "$RCCL_TESTS_MPI_DIR/${coll}_perf" -b 8 -e 1M -f 2 -g 1 -n 5 -w 1 -c 0 \
          > "$d.out" 2>&1
      echo "$coll rc=$?"
    done ) >> "$STATE" 2>&1
  say "debug $1 done"
}

debug 7.14
debug 10.1
timing 7.14 default ""
timing 10.1 default ""
timing 10.1 nodda   "RCCL_DDA_ENABLE=0"

( export ROCM_STACK=10.1; source $BR/common/env.sh
  $PY $BR/rocm10.1/analyze_rccl_why_1node.py "$OUT" ) > "$OUT/analyze.log" 2>&1
say "analyze rc=$?"
echo "$OUT" > "$DONE"
say "DONE"
