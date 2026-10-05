#!/usr/bin/env bash
# 2-node RCCL sweep: node6100 + node6101, one MPI rank per GPU (16 ranks at PPN=8).
# NEW for amd-ubuntu -- amd-cloud had a single node, so its Part B is XGMI-only. This one puts
# the inter-node fabric (8x AMD Pollara "ionic" 400G RoCEv2 NICs per node) in the path.
#
# Run on either node (./launch.sh node6100 rccl-tests/run-rccl-2node.sh). Needs:
#   * $MPI_HOME (Open MPI 5.0.8) and $RCCL_TESTS_MPI_DIR (rccl-tests MPI=1) from amd-software
#   * passwordless ssh between the two nodes (keys in each node's local ~/.ssh)
#
# Knobs (env):
#   HOSTS="node6100.inband node6101.inband"   PPN_LIST="8"  (e.g. "1 2 4 8" for per-node scaling)
#   COLLECTIVES, MIN_BYTES, MAX_BYTES, ITERS, WARMUP, OUT_TAG
#   NET_IF  bootstrap/OOB interface (frontend 10.1.0.0/16)      default eno17695np0
#   IB_HCA  RCCL RDMA devices (the 8 GPU-rail NICs, not mlx5_0) default ionic_0..ionic_7
#   GID     RoCEv2 IPv4-mapped GID index (gid_attrs: 0=link-local, 1=IPv4)  default 1
#   EXTRA_ENV="K=V K=V"  extra NCCL_/RCCL_ settings exported to every rank
#   NUMA_BIND=1  bind each rank to the NUMA node of its GPU/rail (local rank r -> GPU r -> ionic_r;
#                rails 0-3 on NUMA0, 4-7 on NUMA1) via numa-bind.sh; 0 = no binding
# GPU lock: the caller must hold /dev/shm/shaohao-gpu.lock on BOTH nodes (run_part_b_2node.sh does).
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
BIN=$RCCL_TESTS_MPI_DIR
HOSTS=(${HOSTS:-node6100.inband node6101.inband})
PPN_LIST="${PPN_LIST:-8}"
COLLECTIVES="${COLLECTIVES:-all_reduce all_gather reduce_scatter alltoall broadcast sendrecv}"
MIN_BYTES="${MIN_BYTES:-8}"; MAX_BYTES="${MAX_BYTES:-16G}"; STEP_FACTOR="${STEP_FACTOR:-2}"
ITERS="${ITERS:-20}"; WARMUP="${WARMUP:-5}"
NET_IF="${NET_IF:-eno17695np0}"
IB_HCA="${IB_HCA:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
GID="${GID:-1}"
NUMA_BIND="${NUMA_BIND:-1}"
read -r -a EXTRA <<<"${EXTRA_ENV:-}"   # was a bare $EXTRA_ENV: unbound under set -u -> abort in 1 s
TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/rccl/rccl_2node_${OUT_TAG:-all}_$TS; mkdir -p "$OUT"
SUM=$OUT/rccl_summary.txt

[[ -x $MPI_HOME/bin/mpirun ]] || { echo "ERROR: no mpirun at $MPI_HOME (amd-software/setup/build-openmpi.sh)" >&2; exit 1; }
[[ -d $BIN ]] || { echo "ERROR: no MPI rccl-tests at $BIN (amd-software/setup/build-rccl-tests.sh)" >&2; exit 1; }
assert_gpus_idle
for h in "${HOSTS[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" true || { echo "ERROR: cannot ssh to $h" >&2; exit 1; }
  b=$(ssh -o BatchMode=yes "$h" "rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print \$NF}' | grep -cv '^0\$'")
  [[ "${b:-0}" -eq 0 ]] || echo "WARNING: $b GPU(s) busy on $h"
done

RCCL_ENV=(NCCL_IB_DISABLE=0 NCCL_IB_HCA="$IB_HCA" NCCL_IB_GID_INDEX="$GID"
          NCCL_SOCKET_IFNAME="$NET_IF" NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}" "${EXTRA[@]}")
XARGS=(-x PATH -x LD_LIBRARY_PATH)
for kv in "${RCCL_ENV[@]}"; do XARGS+=(-x "$kv"); done
LAUNCH=(); [[ $NUMA_BIND == 1 ]] && LAUNCH=("$(cd "$(dirname "$0")" && pwd)/numa-bind.sh")

{ echo "RCCL 2-node sweep $TS on ${HOSTS[*]}"; echo "bins: $BIN"; echo "mpi : $MPI_HOME";
  echo "env : ${RCCL_ENV[*]} NCCL_DEBUG=${NCCL_DEBUG:-WARN} (INFO only in the nccl_info_* probe)"; echo "bind: ${LAUNCH[*]:-none}"; echo;
  printf '%-14s %4s %5s %12s %14s %14s\n' collective PPN ranks max_size busbw_GB/s lat_8B_us; } | tee "$SUM"

for PPN in $PPN_LIST; do
  NP=$(( PPN * ${#HOSTS[@]} ))
  hostlist=$(printf "%s:$PPN," "${HOSTS[@]}"); hostlist=${hostlist%,}
  for coll in $COLLECTIVES; do
    exe="$BIN/${coll}_perf"
    [[ -x "$exe" ]] || { echo "SKIP $coll (no binary)" | tee -a "$SUM"; continue; }
    maxb="$MAX_BYTES"; [[ "$coll" == alltoall* ]] && maxb="${ALLTOALL_MAX:-4G}"
    log="$OUT/${coll}_ppn${PPN}_np${NP}.log"
    dbg=(-x NCCL_DEBUG="${NCCL_DEBUG:-WARN}")
    # Transport probe before the first point: a short run at NCCL_DEBUG=INFO into its own log, so
    # the log records which NICs/transport RCCL chose. Not on the measured run: INFO prints a line
    # per collective call into stdout, which corrupts the result rows and perturbs the timing.
    if [[ $coll == "${COLLECTIVES%% *}" && $PPN == "${PPN_LIST%% *}" ]]; then
      info="$OUT/nccl_info_${coll}.p${PPN}.log"   # name must not match *_ppn*_np*.log (analyzer)
      timeout 600 "$MPI_HOME/bin/mpirun" --prefix "$MPI_HOME" -np "$NP" --host "$hostlist" \
          --map-by "ppr:$PPN:node" --bind-to none --mca plm_rsh_args "-o BatchMode=yes" \
          --mca btl tcp,self --mca btl_tcp_if_include "$NET_IF" --mca oob_tcp_if_include "$NET_IF" \
          "${XARGS[@]}" -x NCCL_DEBUG=INFO \
          "${LAUNCH[@]}" "$exe" -b 8 -e 64M -f 8 -g 1 -n 2 -w 1 -c 1 >"$info" 2>&1
      { echo "transport: $(grep -a -m1 -oE 'NET/IB : Using.{0,200}' "$info" || echo 'no NET/IB line')"
        echo "channels : $(grep -a -oE 'via NET/[A-Za-z-]+(/[0-9]+)?(/GDRDMA)?' "$info" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
        grep -a -q 'via NET/Socket' "$info" && echo "WARNING: NET/Socket fallback (TCP), not RoCE"; } | tee -a "$SUM"
    fi
    timeout 1800 "$MPI_HOME/bin/mpirun" --prefix "$MPI_HOME" -np "$NP" --host "$hostlist" \
        --map-by "ppr:$PPN:node" --bind-to none \
        --mca plm_rsh_args "-o BatchMode=yes" \
        --mca btl tcp,self --mca btl_tcp_if_include "$NET_IF" --mca oob_tcp_if_include "$NET_IF" \
        "${XARGS[@]}" "${dbg[@]}" \
        "${LAUNCH[@]}" "$exe" -b "$MIN_BYTES" -e "$maxb" -f "$STEP_FACTOR" -g 1 -n "$ITERS" -w "$WARMUP" -c 1 \
        >"$log" 2>&1
    rc=$?
    # data rows: last row -> in-place busbw ($(NF-1)); first row -> in-place time (us, $(NF-3))
    read -r size busbw lat < <(awk '/^ *[0-9]+ +[0-9]+ +[a-z]/ {if(!l) l=$(NF-3); sz=$1; bw=$(NF-1)} END {print sz, bw, l}' "$log")
    printf '%-14s %4s %5s %12s %14s %14s   rc=%s\n' "$coll" "$PPN" "$NP" "${size:--}" "${busbw:--}" "${lat:--}" "$rc" | tee -a "$SUM"
  done
done
echo "results: $OUT" | tee -a "$SUM"
echo "$OUT" > "$LOG_ROOT/rccl/CURRENT_2NODE_DIR.txt"
