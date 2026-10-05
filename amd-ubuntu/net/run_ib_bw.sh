#!/usr/bin/env bash
# Host-level RDMA bandwidth between node6100 and node6101 on each ionic rail (no GPU needed).
# Baseline for rccl-tests/run_part_b_2node.sh: what each 400G RoCEv2 rail delivers by itself,
# then all 8 rails at once. perftest ib_write_bw, server on node6101, client on node6100.
#
#   ./launch.sh node6100 net/run_ib_bw.sh            # or run directly on node6100
# Knobs: PEER (node6101), GID (1 = RoCEv2 IPv4), DUR (s per test), SIZE (bytes), QPS,
#        NUMA=1 binds each ib_write_bw to its NIC's NUMA node (CPU + memory) on both ends,
#        STAGES ("single all8")
# Output: logs/<node>/net/ib_bw_<ts>/*.log, results/<node>/ib_bw.md
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
PEER=${PEER:-node6101}; GID=${GID:-1}; DUR=${DUR:-10}; SIZE=${SIZE:-1048576}; QPS=${QPS:-4}
RAILS=(ionic_0 ionic_1 ionic_2 ionic_3 ionic_4 ionic_5 ionic_6 ionic_7)
OUT=$LOG_ROOT/net/ib_bw_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT" "$RESULTS"
peer() { ssh -o BatchMode=yes "$PEER.inband" "$@"; }
ARGS="-x $GID -F --report_gbits -D $DUR -s $SIZE -q $QPS"
NUMA=${NUMA:-1}; STAGES=${STAGES:-single all8}
# numactl prefix for rail $1 (NIC NUMA node is the same on both nodes).
bind() { [[ $NUMA == 1 ]] || return 0
         local n; n=$(cat /sys/class/infiniband/$1/device/numa_node); echo "numactl --cpunodebind=$n --membind=$n"; }

# $1 = list of rails run concurrently, $2 = tag
run() {
  local tag=$2 i=0 r port pids=()
  for r in $1; do
    port=$((18515 + i)); i=$((i + 1))
    peer "nohup $(bind $r) ib_write_bw -d $r -p $port $ARGS > /tmp/$(id -un)-ibbw-$r.log 2>&1 &"
  done
  sleep 2; i=0
  for r in $1; do
    port=$((18515 + i)); i=$((i + 1))
    $(bind "$r") ib_write_bw -d "$r" -p "$port" $ARGS "$PEER.inband" > "$OUT/${tag}_$r.log" 2>&1 & pids+=($!)
  done
  wait "${pids[@]}"
  peer "rm -f /tmp/$(id -un)-ibbw-*.log"
}

[[ " $STAGES " == *" single "* ]] && for r in "${RAILS[@]}"; do run "$r" single; done
[[ " $STAGES " == *" all8 "* ]] && run "${RAILS[*]}" all8

# BW average [Gb/s] is the 4th column of the result line under the "#bytes" header.
bw() { awk '/#bytes/{getline; print $4}' "$1" 2>/dev/null; }
{
  echo "# RDMA write bandwidth, $HOSTNAME_S → $PEER (ionic RoCEv2)"; echo
  echo "perftest \`ib_write_bw $ARGS\`, NUMA binding=$NUMA, $(date -Iseconds). Line rate 400 Gb/s per rail, 3,200 Gb/s (400 GB/s) per node."
  echo "Raw logs: \`${OUT#$BENCH_ROOT/}\`."; echo
  echo "| rail | alone (Gb/s) | all 8 at once (Gb/s) | % of 400 alone |"; echo "|---|---:|---:|---:|"
  tot=0
  for r in "${RAILS[@]}"; do
    a=$(bw "$OUT/single_$r.log"); b=$(bw "$OUT/all8_$r.log")
    pct=$(awk -v a="${a:-0}" 'BEGIN{printf "%.0f%%", a/4}')
    echo "| $r | ${a:-FAIL} | ${b:-FAIL} | $pct |"
    tot=$(awk -v t="$tot" -v b="${b:-0}" 'BEGIN{print t+b}')
  done
  awk -v t="$tot" 'BEGIN{printf "| **total, 8 rails** | | **%.1f** (%.1f GB/s) | %.0f%% of 3,200 |\n", t, t/8, t/32}'
} > "$RESULTS/ib_bw.md"
cat "$RESULTS/ib_bw.md"
