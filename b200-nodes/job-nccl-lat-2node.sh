#!/bin/bash
# Inter-node latency between two B200 nodes: host RDMA (perftest) -> NCCL small messages.
# Apple-to-apple twin of ../amd-ubuntu/rccl-tests/run_rccl_lat_2node.sh (MI355X node6100+6101):
# same tests, sizes, iterations and layout, so the two result files can be read side by side.
#   A. ib_write_lat / ib_send_lat, host memory, each of the 8 GPU rails alone, -a -n 5000,
#      both ends bound to the NIC's NUMA node
#   B. nccl-tests 8 B..4 MB (factor 2), 1000 iterations + 100 warm-up, no data check,
#      one MPI rank per GPU (as on the AMD side; the bandwidth job uses 1 rank x 8 GPUs),
#      each rank bound to its GPU's NUMA node:
#        2 nodes x 1 GPU : sendrecv, all_reduce             (GPU-NIC-wire-NIC-GPU)
#        2 nodes x 8 GPUs: sendrecv all_reduce all_gather reduce_scatter broadcast alltoall
#        1 node  x 8 GPUs: the same six (NVLink only, reference)
#   C. analysis with the AMD analyzer -> out-nccl-lat/<job>/nccl_lat.md
#
# Submit:  sbatch job-nccl-lat-2node.sh            (any two healthy B200 nodes)
#          sbatch -w node5600-c1,node5601-c1 job-nccl-lat-2node.sh
# node5802-c1 is excluded: its routes to node5602-c1/node5702-c1 run at half bandwidth
# (out-nccl-2node/summary.md).
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:b200:8
#SBATCH --exclude=node5802-c1
#SBATCH --mem=200GB
#SBATCH -t 60
#SBATCH -J nccl-lat-2node
#SBATCH --exclusive
#SBATCH -o out-nccl-lat/%x-%J.out

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
OUT=$PWD/out-nccl-lat/$SLURM_JOB_ID; mkdir -p "$OUT"
NCCL_DIR=/orcd/data/orcd/022/benchmarks/nccl-tests
BIN=$NCCL_DIR/build-nvhpc-26.1
ANALYZE=/orcd/data/orcd/022/benchmarks/amd-ubuntu/rccl-tests/analyze_rccl_lat.py

module purge
module load nvhpc/26.1
NVHPC_HOME=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
OMPI_HOME=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi
export LD_LIBRARY_PATH=$OMPI_HOME/lib:$LD_LIBRARY_PATH

RAILS=(mlx5_4 mlx5_7 mlx5_8 mlx5_9 mlx5_10 mlx5_13 mlx5_14 mlx5_15)
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SRV=${NODES[0]}; CLI=${NODES[1]}
echo "nccl latency, job $SLURM_JOB_ID, $(date -Iseconds), nodes $SRV $CLI"

# ---- A. host RDMA latency ------------------------------------------------------------
LAT_ARGS="-F -a -n ${IB_ITERS:-5000}"
on() {  # on <node> <rail> <cmd...>: run on node, bound to the rail's NUMA node if numactl exists
   local n=$1 r=$2; shift 2
   srun --overlap -N1 -n1 -w "$n" bash -c \
     'nn=$(cat /sys/class/infiniband/'"$r"'/device/numa_node 2>/dev/null)
      if command -v numactl >/dev/null && [[ ${nn:--1} -ge 0 ]]; then exec numactl --cpunodebind=$nn --membind=$nn "$@"; else exec "$@"; fi' _ "$@"
}
for t in write send; do
   for r in "${RAILS[@]}"; do
      on "$SRV" "$r" ib_${t}_lat -d "$r" -p 18600 $LAT_ARGS > "$OUT/ib_${t}_$r.srv" 2>&1 &
      spid=$!; sleep 4
      on "$CLI" "$r" ib_${t}_lat -d "$r" -p 18600 $LAT_ARGS "$SRV" > "$OUT/${t}_$r.log" 2>&1
      wait $spid 2>/dev/null
   done
done
col() { awk -v s="$2" -v c="$3" '$1==s && $2 ~ /^[0-9]+$/ {print $c; exit}' "$1" 2>/dev/null; }
{
   echo "# RDMA latency, $SRV ↔ $CLI (NDR InfiniBand, one rail at a time)"; echo
   echo "perftest \`ib_<test>_lat $LAT_ARGS\`, NUMA-bound, $(date -Iseconds). One-way latency in µs (half of the round trip)."; echo
   for t in write send; do
      echo "## ib_${t}_lat — median (t_typical) per rail, µs"; echo
      printf '| bytes |'; for r in "${RAILS[@]}"; do printf ' %s |' "$r"; done; echo
      printf '|---:|'; for r in "${RAILS[@]}"; do printf -- '---:|'; done; echo
      for s in 2 8 64 512 4096 65536 1048576; do
         printf '| %s |' "$s"; for r in "${RAILS[@]}"; do printf ' %s |' "$(col "$OUT/${t}_$r.log" $s 5)"; done; echo
      done
      echo; echo "Tail at 8 bytes (${RAILS[0]}): avg $(col "$OUT/${t}_${RAILS[0]}.log" 8 6), 99% $(col "$OUT/${t}_${RAILS[0]}.log" 8 8), 99.9% $(col "$OUT/${t}_${RAILS[0]}.log" 8 9) µs."; echo
   done
} > "$OUT/ib_lat.md"
cat "$OUT/ib_lat.md"

# ---- B. NCCL small-message latency ------------------------------------------------------
# per-rank NUMA binding (the twin of amd-ubuntu/rccl-tests/numa-bind.sh): local rank r -> GPU r
BIND=$OUT/numa-bind-gpu.sh
cat > "$BIND" <<'EOF'
#!/bin/bash
r=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$r" 2>/dev/null | tr 'A-F' 'a-f')
nn=$(cat /sys/bus/pci/devices/${bus#0000}/numa_node 2>/dev/null)
if command -v numactl >/dev/null && [[ ${nn:--1} -ge 0 ]]; then exec numactl --cpunodebind=$nn --membind=$nn "$@"; fi
exec "$@"
EOF
chmod +x "$BIND"

export NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_SOCKET_IFNAME=^lo,docker NCCL_DEBUG=WARN
export NCCL_IB_HCA=$(IFS=,; echo "${RAILS[*]}")
XENV="-x NCCL_IB_DISABLE -x NCCL_NET_GDR_LEVEL -x NCCL_IB_HCA -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG -x PATH -x LD_LIBRARY_PATH"
MPI_FLAGS="--mca pml ob1 --mca btl tcp,self --mca coll_ucc_enable 0 --mca coll_hcoll_enable 0 \
   --mca btl_openib_warn_no_device_params_found 0 --bind-to none"
COLLS="sendrecv all_reduce all_gather reduce_scatter broadcast alltoall"
RUNS=$OUT/rccl_runs.txt; : > "$RUNS"

# $1 tag  $2 "host:ppn,..."  $3 PPN  $4 collectives
nccl() {
   local d=$OUT/$1 np=$(( $3 * $(tr ',' '\n' <<<"$2" | wc -l) )); mkdir -p "$d"
   # transport probe (INFO) into its own log, as on the AMD side
   mpirun -np $np --host "$2" --map-by ppr:$3:node $MPI_FLAGS $XENV -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET \
      "$BIND" $BIN/${4%% *}_perf -b 8 -e 64M -f 8 -g 1 -n 2 -w 1 -c 1 > "$d/nccl_info.log" 2>&1
   for c in $4; do
      timeout 1200 mpirun -np $np --host "$2" --map-by ppr:$3:node $MPI_FLAGS $XENV \
         "$BIND" $BIN/${c}_perf -b 8 -e 4M -f 2 -g 1 -n ${ITERS:-1000} -w ${WARMUP:-100} -c 0 \
         > "$d/${c}_ppn$3_np$np.log" 2>&1
      echo "$1 $c rc=$? 8B: $(awk '/^ *8 +[0-9]+ +[a-z]/ {print $6" us"; exit}' "$d/${c}_ppn$3_np$np.log")"
   done
   echo "$1|${2//:[0-9]/}|$3||$d" | tr ',' ' ' >> "$RUNS"
}
nccl 2n_ppn1 "$SRV:1,$CLI:1" 1 "sendrecv all_reduce"
nccl 2n_ppn8 "$SRV:8,$CLI:8" 8 "$COLLS"
nccl 1n_ppn8 "$SRV:8"        8 "$COLLS"

# ---- C. analysis --------------------------------------------------------------------------
python3 "$ANALYZE" "$RUNS" -o "$OUT/nccl_lat.md" --title "NCCL latency, $SRV + $CLI (B200)" \
   --tool nccl-tests --intra NVLink
grep -h "NCCL version\|NET/IB : Using" "$OUT"/2n_ppn8/nccl_info.log | head -2
echo "done $(date -Iseconds): $OUT/{ib_lat,nccl_lat}.md"
