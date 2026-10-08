#!/usr/bin/env bash
# Why is 2-node sendrecv only 57% of one rail (28.6 of 50 GB/s) while the other
# collectives reach 92-95% of 400 GB/s? Uses BOTH nodes; run from node6100:
#   ./launch.sh node6100 rccl-tests/run_sendrecv_check.sh
#
# A. ib_write_bw, host memory, 1 MiB, node6100 -> node6101 (perftest here has no ROCm support):
#    same rail ionic_0 -> ionic_0 with 1 / 2 / 4 queue pairs (QPs), and cross rail
#    ionic_7 -> ionic_0 (the path of the 16-rank sendrecv pair GPU7@6100 -> GPU0@6101).
#    1 QP well below 400 Gb/s -> a single connection cannot fill the rail.
#    cross rail slow -> the pair's traffic leaves its rail (B200's NCCL avoids that with PXN).
# B. rccl-tests sendrecv, 1 MiB..16 GiB, one RCCL setting per run (EXTRA_ENV).
# Then analyze_sendrecv_check.py writes results/ubuntu/sendrecv-check.md.
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rccl/sendrecv_check_$TS; mkdir -p "$OUT"
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

# ---- A. raw RDMA, one rail ------------------------------------------------------
numa() { echo "numactl --cpunodebind=$(cat /sys/class/infiniband/$1/device/numa_node) --membind=$(cat /sys/class/infiniband/$1/device/numa_node)"; }
ib() {   # $1 tag  $2 local dev  $3 peer dev  $4 QPs
  local port=18600 args="-x 1 -F --report_gbits -D 10 -s 1048576 -q $4"
  ssh -o BatchMode=yes "$PEER" "cd /; timeout 60 numactl --cpunodebind=\$(cat /sys/class/infiniband/$3/device/numa_node) \
       ib_write_bw -d $3 -p $port $args > $OUT/ib_$1.server.log 2>&1 < /dev/null &"
  sleep 3
  timeout 60 $(numa "$2") ib_write_bw -d "$2" -p $port $args "$PEER" > "$OUT/ib_$1.log" 2>&1
  say "ib $1 ($2 -> $3, $4 QP): rc=$? $(awk '/#bytes/{getline; print $4" Gb/s"}' "$OUT/ib_$1.log")"
  sleep 2
}
ib same_q1  ionic_0 ionic_0 1
ib same_q2  ionic_0 ionic_0 2
ib same_q4  ionic_0 ionic_0 4
ib cross_q1 ionic_7 ionic_0 1
ib cross_q4 ionic_7 ionic_0 4

# ---- B. RCCL sendrecv, one setting per run ----------------------------------------
rccl() {  # $1 tag  $2 PPN  $3 EXTRA_ENV
  rm -f "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"
  PPN_LIST=$2 COLLECTIVES=sendrecv MIN_BYTES=1M MAX_BYTES=16G OUT_TAG=srchk_$1 EXTRA_ENV="$3" \
      ./run-rccl-2node.sh > "$OUT/rccl_$1.log" 2>&1
  local d; d=$(cat "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt" 2>/dev/null)
  echo "$1|$2|$3|$d" >> "$OUT/rccl_runs.txt"
  say "rccl $1 (PPN=$2, $3): $(awk '$1=="sendrecv"{print $5" GB/s"}' "$d/rccl_summary.txt" 2>/dev/null)"
}
rccl default_ppn1 1 ""
rccl default      8 ""
rccl nch4         8 "NCCL_NCHANNELS_PER_NET_PEER=4"
rccl nch8         8 "NCCL_NCHANNELS_PER_NET_PEER=8"
rccl qps2         8 "NCCL_IB_QPS_PER_CONNECTION=2"
rccl qps4         8 "NCCL_IB_QPS_PER_CONNECTION=4"
rccl chunk1m      8 "NCCL_P2P_NET_CHUNKSIZE=1048576"
rccl pxn          8 "NCCL_PXN_DISABLE=0 NCCL_P2P_PXN_LEVEL=2"
rccl nch8_qps4    8 "NCCL_NCHANNELS_PER_NET_PEER=8 NCCL_IB_QPS_PER_CONNECTION=4"

# release locks before the analysis
eval "exec ${PEERLOCK[1]}>&-"; flock -u 9
say "runs done, locks released"
python3 analyze_sendrecv_check.py "$OUT" && say "wrote $RESULTS/../ubuntu/sendrecv-check.md"
say "done"
