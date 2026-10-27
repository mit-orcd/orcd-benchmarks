#!/usr/bin/env bash
# ROCm 10.1 copy of ../rccl-tests/run_rccl_lat_1node.sh: intra-node RCCL latency, 1 node x 8 GPUs,
# same tests, sizes, iterations, settings and order. Differences:
#   * does not run compare_lat_1node.py (the existing comparison md files stay as they are);
#   * skips the GPU lock when the caller already holds it (GPU_LOCKS_HELD=1).
#   ./launch.sh node6100 with_rocm.sh 10.1 rocm10.1/run_rccl_lat_1node_rocm10.1.sh
# Results: $OUT/STATE.txt, $OUT/rccl_runs.txt, results/<node>/rocm10.1/rccl_lat_1node.md
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd /orcd/data/orcd/022/benchmarks/amd-ubuntu/rccl-tests
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rccl/lat_1node_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "1-node latency run start on $HOSTNAME_S, ROCM_STACK=${ROCM_STACK:-host} ($OUT)"

LOCK=/dev/shm/shaohao-gpu.lock
if [[ -z ${GPU_LOCKS_HELD:-} ]]; then
  say "waiting for $LOCK"
  exec 9>>"$LOCK"; flock 9
  say "got $HOSTNAME_S lock"
  export GPU_LOCKS_HELD=1
fi

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
