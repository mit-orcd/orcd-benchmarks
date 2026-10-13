#!/usr/bin/env bash
# Host-level RDMA latency between node6100 and node6101 on each ionic rail (no GPU needed).
# Baseline for rccl-tests/run_rccl_lat_2node.sh: the wire + NIC latency with no GPU or RCCL in
# the path. perftest ib_write_lat and ib_send_lat, one rail at a time, server on node6101,
# client on node6100, each side bound to its NIC's NUMA node.
#
#   ./launch.sh node6100 net/run_ib_lat.sh       # or called by rccl-tests/run_rccl_lat_2node.sh
# Knobs: PEER (node6101), GID (1 = RoCEv2 IPv4), ITERS (per size), TESTS ("write send"),
#        RAILS ("ionic_0 ... ionic_7"), NUMA=1
# Output: logs/<node>/net/ib_lat_<ts>/*.log, results/<node>/ib_lat.md
# perftest prints t_typical (median), t_avg and the 99% / 99.9% tails in usec; with -a the
# reported latency is one-way (half the round trip) for send_lat and write_lat.
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
PEER=${PEER:-node6101}; GID=${GID:-1}; ITERS=${ITERS:-5000}; TESTS=${TESTS:-write send}; NUMA=${NUMA:-1}
read -r -a RAILS <<<"${RAILS:-ionic_0 ionic_1 ionic_2 ionic_3 ionic_4 ionic_5 ionic_6 ionic_7}"
OUT=$LOG_ROOT/net/ib_lat_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT" "$RESULTS"
peer() { ssh -o BatchMode=yes "$PEER.inband" "$@"; }
ARGS="-x $GID -F -a -n $ITERS"
bind() { [[ $NUMA == 1 ]] || return 0
         local n; n=$(cat /sys/class/infiniband/$1/device/numa_node); echo "numactl --cpunodebind=$n --membind=$n"; }

for t in $TESTS; do
  for r in "${RAILS[@]}"; do
    peer "nohup $(bind $r) ib_${t}_lat -d $r -p 18600 $ARGS > /tmp/$(id -un)-iblat.log 2>&1 &"
    sleep 2
    timeout 600 $(bind "$r") "ib_${t}_lat" -d "$r" -p 18600 $ARGS "$PEER.inband" > "$OUT/${t}_$r.log" 2>&1
    peer "pkill -u $(id -un) -f 'ib_${t}_lat -d $r' ; rm -f /tmp/$(id -un)-iblat.log" 2>/dev/null
  done
done

# result rows: bytes iters t_min t_max t_typical t_avg t_stdev 99% 99.9%
col() { awk -v s="$2" -v c="$3" '$1==s && $2 ~ /^[0-9]+$/ {print $c; exit}' "$1" 2>/dev/null; }
SIZES="2 8 64 512 4096 65536 1048576"
{
  echo "# RDMA latency, $HOSTNAME_S → $PEER (ionic RoCEv2, one rail at a time)"; echo
  echo "perftest \`ib_<test>_lat $ARGS\`, NUMA binding=$NUMA, $(date -Iseconds)."
  echo "One-way latency in µs (half of the round trip). Raw logs: \`${OUT#$BENCH_ROOT/}\`."; echo
  for t in $TESTS; do
    echo "## ib_${t}_lat — median (t_typical) per rail, µs"; echo
    printf '| bytes |'; for r in "${RAILS[@]}"; do printf ' %s |' "${r#ionic_}"; done; echo
    printf '|---:|'; for r in "${RAILS[@]}"; do printf -- '---:|'; done; echo
    for s in $SIZES; do
      printf '| %s |' "$s"
      for r in "${RAILS[@]}"; do printf ' %s |' "$(col "$OUT/${t}_$r.log" "$s" 5)"; done; echo
    done
    echo; echo "Tail at 8 bytes (rail 0): avg $(col "$OUT/${t}_${RAILS[0]}.log" 8 6), 99% $(col "$OUT/${t}_${RAILS[0]}.log" 8 8), 99.9% $(col "$OUT/${t}_${RAILS[0]}.log" 8 9) µs."; echo
  done
} > "$RESULTS/ib_lat.md"
cat "$RESULTS/ib_lat.md"
