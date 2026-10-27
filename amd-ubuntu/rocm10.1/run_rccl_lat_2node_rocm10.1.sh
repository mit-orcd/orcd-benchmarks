#!/usr/bin/env bash
# ROCm 10.1 copy of ../rccl-tests/run_rccl_lat_2node.sh: inter-node RCCL latency, node6100 <-> node6101,
# same tests, sizes, iterations and settings as the ROCm 7.14 run (default + NCCL_NET=IB). Differences:
#   * no perftest part (host RDMA does not use ROCm; not affected by the update);
#   * skips the GPU locks when the caller already holds both (GPU_LOCKS_HELD=1).
# Run from node6100:
#   ./launch.sh node6100 with_rocm.sh 10.1 rocm10.1/run_rccl_lat_2node_rocm10.1.sh
# Results: $OUT/STATE.txt, $OUT/rccl_runs.txt, results/node6100/rocm10.1/rccl_lat.md
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd /orcd/data/orcd/022/benchmarks/amd-ubuntu/rccl-tests
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rccl/lat_2node_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "latency run start, ROCM_STACK=${ROCM_STACK:-host} ($OUT)"

# ---- GPU locks: node6100 first, then node6101 (same as run_part_b_2node.sh) ----
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

# ---- B. RCCL small-message latency ------------------------------------------------------
# $1 tag  $2 hosts  $3 PPN  $4 collectives  $5 EXTRA_ENV
rccl() {
  rm -f "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"
  HOSTS="$2" PPN_LIST=$3 COLLECTIVES="$4" MIN_BYTES=8 MAX_BYTES=4M ALLTOALL_MAX=4M \
    ITERS=${ITERS:-1000} WARMUP=${WARMUP:-100} CHECK=0 OUT_TAG=lat_$1 EXTRA_ENV="$5" \
    ./run-rccl-2node.sh > "$OUT/rccl_$1.log" 2>&1
  local d; d=$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt" 2>/dev/null)
  echo "$1|$2|$3|$5|$d" >> "$OUT/rccl_runs.txt"
  say "rccl $1 (PPN=$3, ${5:-default}): $(awk '$1!="collective" && $2 ~ /^[0-9]+$/ {printf "%s %sus; ", $1, $6}' "$d/rccl_summary.txt" 2>/dev/null)"
}
TWO="node6100.inband node6101.inband"; ONE="node6100.inband"
COLLS="sendrecv all_reduce all_gather reduce_scatter broadcast alltoall"
SETTINGS=("default|")
SETTINGS+=("netib|NCCL_NET=IB")   # as the ROCm 7.14 run
for s in "${SETTINGS[@]}"; do
  tag=${s%%|*}; env=${s#*|}
  say "B. RCCL latency, setting $tag"
  rccl ${tag}_2n_ppn1 "$TWO" 1 "sendrecv all_reduce"  "$env"   # 1 GPU per node: GPU-NIC-wire-NIC-GPU
  rccl ${tag}_2n_ppn8 "$TWO" 8 "$COLLS"               "$env"   # 16 GPUs, as the bandwidth tables
  rccl ${tag}_1n_ppn8 "$ONE" 8 "$COLLS"               "$env"   # 8 GPUs, XGMI only (reference)
done

# ---- C. analysis --------------------------------------------------------------------------
$PY analyze_rccl_lat.py "$OUT/rccl_runs.txt" -o "$RESULTS/rccl_lat.md" > "$OUT/analyze.log" 2>&1
say "C. analyze rc=$? -> $RESULTS/rccl_lat.md"
say "latency run DONE"
