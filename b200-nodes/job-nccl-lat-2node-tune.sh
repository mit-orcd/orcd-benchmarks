#!/bin/bash
# B200 2 nodes x 8 GPUs NCCL latency with fixed NCCL_ALGO / NCCL_PROTO.
# Question: at 1 MB B200 is much slower across 2 nodes than in 1 node (all_gather 18 -> 83 us,
# slower than at 4 MB), which gives MI355X its 1 MB wins. Is that NCCL's default algorithm /
# protocol choice? Same tests as job-nccl-lat-2node.sh part B (8 B..4 MB, 1000 iters + 100 warm-up,
# one rank per GPU, NUMA-bound), run once per fixed setting; default twice (start, end) plus
# 1 node x 8 GPUs default on the same node (for the network part).
# A TUNING log (NCCL_DEBUG_SUBSYS=TUNING) shows which algorithm/protocol the default picks.
# Analysis: ../amd-ubuntu/rccl-tests/b200_lat_tune.py (run by the post job).
#
# Submit:  sbatch job-nccl-lat-2node-tune.sh
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:b200:8
#SBATCH --exclude=node5802-c1
#SBATCH --mem=200GB
#SBATCH -t 150
#SBATCH -J nccl-lat-2node-tune
#SBATCH --exclusive
#SBATCH -o out-nccl-lat-tune/%x-%J.out

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
OUT=$PWD/out-nccl-lat-tune/$SLURM_JOB_ID; mkdir -p "$OUT"
BIN=/orcd/data/orcd/022/benchmarks/nccl-tests/build-nvhpc-26.1
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

module purge
module load nvhpc/26.1
NVHPC_HOME=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
OMPI_HOME=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi
export LD_LIBRARY_PATH=$OMPI_HOME/lib:$LD_LIBRARY_PATH

RAILS=(mlx5_4 mlx5_7 mlx5_8 mlx5_9 mlx5_10 mlx5_13 mlx5_14 mlx5_15)
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SRV=${NODES[0]}; CLI=${NODES[1]}
say "nccl latency tuning, job $SLURM_JOB_ID, nodes $SRV $CLI"

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
BASEX="-x NCCL_IB_DISABLE -x NCCL_NET_GDR_LEVEL -x NCCL_IB_HCA -x NCCL_SOCKET_IFNAME -x PATH -x LD_LIBRARY_PATH"
MPI_FLAGS="--mca pml ob1 --mca btl tcp,self --mca coll_ucc_enable 0 --mca coll_hcoll_enable 0 \
   --mca btl_openib_warn_no_device_params_found 0 --bind-to none"
COLLS="sendrecv all_reduce all_gather reduce_scatter broadcast alltoall"
RUNS=$OUT/rccl_runs.txt; : > "$RUNS"

# $1 setting tag  $2 "K=V K=V" (empty = default)  $3 hosts  $4 ppn  $5 layout
nccl() {
   local d=$OUT/$1_$5 np=$(( $4 * $(tr ',' '\n' <<<"$3" | wc -l) )) xenv="$BASEX -x NCCL_DEBUG" kv c; mkdir -p "$d"
   for kv in $2; do export "$kv"; xenv+=" -x ${kv%%=*}"; done
   for c in $COLLS; do
      timeout 900 mpirun -np $np --host "$3" --map-by ppr:$4:node $MPI_FLAGS $xenv \
         "$BIND" $BIN/${c}_perf -b 8 -e 4M -f 2 -g 1 -n ${ITERS:-1000} -w ${WARMUP:-100} -c 0 \
         > "$d/${c}_ppn$4_np$np.log" 2>&1
   done
   for kv in $2; do unset "${kv%%=*}"; done
   echo "$1_$5|${3//:[0-9]/}|$4|${2:-default}|$d" | tr ',' ' ' >> "$RUNS"
   say "$1 $5 (${2:-default}): 1 MB: $(for c in $COLLS; do printf '%s %sus; ' $c "$(awk '$1==1048576 {print $6; exit}' "$d/${c}_ppn$4_np$np.log")"; done)"
}
H2="$SRV:8,$CLI:8"
# which algorithm/protocol the default picks, 64 KB..4 MB
for c in all_gather reduce_scatter broadcast all_reduce; do
   mpirun -np 16 --host "$H2" --map-by ppr:8:node $MPI_FLAGS $BASEX -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING \
      "$BIND" $BIN/${c}_perf -b 64K -e 4M -f 4 -g 1 -n 5 -w 2 -c 0 > "$OUT/tuning_${c}.log" 2>&1
done
say "tuning logs written"

nccl r1         ""                               "$H2" 8 2n_ppn8
nccl ll         "NCCL_PROTO=LL"                  "$H2" 8 2n_ppn8
nccl ll128      "NCCL_PROTO=LL128"               "$H2" 8 2n_ppn8
nccl simple     "NCCL_PROTO=Simple"              "$H2" 8 2n_ppn8
nccl ring       "NCCL_ALGO=Ring"                 "$H2" 8 2n_ppn8
nccl ringll     "NCCL_ALGO=Ring NCCL_PROTO=LL"   "$H2" 8 2n_ppn8
nccl ringll128  "NCCL_ALGO=Ring NCCL_PROTO=LL128" "$H2" 8 2n_ppn8
nccl ringsimple "NCCL_ALGO=Ring NCCL_PROTO=Simple" "$H2" 8 2n_ppn8
nccl tree       "NCCL_ALGO=Tree"                 "$H2" 8 2n_ppn8
nccl treell     "NCCL_ALGO=Tree NCCL_PROTO=LL"   "$H2" 8 2n_ppn8
nccl nvls0      "NCCL_NVLS_ENABLE=0"             "$H2" 8 2n_ppn8
nccl r2         ""                               "$H2" 8 2n_ppn8
nccl r1         ""                               "$SRV:8" 8 1n_ppn8

grep -h "NCCL version" "$OUT"/tuning_all_gather.log | head -1 | tee -a "$STATE"
say "latency run DONE"
