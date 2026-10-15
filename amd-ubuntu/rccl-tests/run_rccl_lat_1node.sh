#!/usr/bin/env bash
# Intra-node latency, 1 node x 8 GPUs (XGMI only), RCCL small messages.
# Checks the 1-node numbers behind section 5 of results/mi355x-vs-b200.md (e.g. all_reduce
# 8 B slower than 1 KB; four of the five 2-node and 1-node logs warned "GPUs busy"): the default setting is run
# three times, and RCCL is also run with a fixed protocol / algorithm, so NCCL's own choice
# can be told apart from the hardware.
# Twin of ../../b200-nodes/job-nccl-lat-1node.sh (B200, NCCL): same tests, sizes, iterations,
# settings and order.
#   ./launch.sh node6100 with_rocm.sh 7.14 rccl-tests/run_rccl_lat_1node.sh
#   ./launch.sh node6101 with_rocm.sh 7.14 rccl-tests/run_rccl_lat_1node.sh
# Waits for this node's GPU lock; each run starts only after rocm-smi shows every GPU at 0 %
# (up to 2 min; the reading is logged either way).
# Results: $OUT/STATE.txt, $OUT/rccl_runs.txt, results/<node>[/rocm7.14]/rccl_lat_1node.md,
#          then compare_lat_1node.py refreshes results/ubuntu/latency-1node.md and §3.2 of
#          results/mi355x-vs-b200.md.
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rccl/lat_1node_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "1-node latency run start on $HOSTNAME_S, ROCM_STACK=${ROCM_STACK:-host} ($OUT)"

LOCK=/dev/shm/shaohao-gpu.lock
say "waiting for $LOCK"
exec 9>>"$LOCK"; flock 9
say "got $HOSTNAME_S lock"
export GPU_LOCKS_HELD=1

gpu_use() { rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {printf "%s ", $NF}'; }
wait_idle() {  # until every GPU reads 0 %, at most 2 min
  local u i
  for i in $(seq 1 24); do
    u=$(gpu_use)
    [[ -n $u ]] && ! tr ' ' '\n' <<<"$u" | grep -qv '^0\?$' && { echo "idle ($u)"; return 0; }
    sleep 5
  done
  echo "NOT idle after 2 min: $u"
}

# $1 tag  $2 EXTRA_ENV  $3 label for the runs file
rccl() {
  local idle; idle=$(wait_idle)
  rm -f "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"
  HOSTS="$HOSTNAME_S.inband" PPN_LIST=8 COLLECTIVES="$COLLS" MIN_BYTES=8 MAX_BYTES=4M ALLTOALL_MAX=4M \
    ITERS=${ITERS:-1000} WARMUP=${WARMUP:-100} CHECK=0 OUT_TAG=lat1n_$1 EXTRA_ENV="$2" \
    ./run-rccl-2node.sh > "$OUT/rccl_$1.log" 2>&1
  local d; d=$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt" 2>/dev/null)
  echo "$1_1n_ppn8|$HOSTNAME_S.inband|8|$3|$d" >> "$OUT/rccl_runs.txt"
  say "rccl $1 (${2:-default}; GPUs before: $idle): $(awk '$1!="collective" && $2 ~ /^[0-9]+$/ {printf "%s %sus; ", $1, $6}' "$d/rccl_summary.txt" 2>/dev/null)"
}
COLLS="sendrecv all_reduce all_gather reduce_scatter broadcast alltoall"
# default three times, spread over the run; fixed protocol / algorithm in between
rccl r1     ""                              "default, run 1"
rccl ll     "NCCL_PROTO=LL"                 "NCCL_PROTO=LL"
rccl ringll "NCCL_ALGO=Ring NCCL_PROTO=LL"  "NCCL_ALGO=Ring NCCL_PROTO=LL"
rccl r2     ""                              "default, run 2"
rccl treell "NCCL_ALGO=Tree NCCL_PROTO=LL"  "NCCL_ALGO=Tree NCCL_PROTO=LL"
rccl r3     ""                              "default, run 3"

$PY analyze_rccl_lat.py "$OUT/rccl_runs.txt" -o "$RESULTS/rccl_lat_1node.md" \
    --title "RCCL latency inside one node, $HOSTNAME_S (MI355X)" > "$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $RESULTS/rccl_lat_1node.md"
say "latency run DONE"
$PY compare_lat_1node.py >> "$OUT/analyze.log" 2>&1
say "compare_lat_1node rc=$?"
