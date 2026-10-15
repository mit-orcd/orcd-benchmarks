#!/bin/bash
# Intra-node latency, 1 node x 8 B200 GPUs (NVLink only), NCCL small messages.
# Checks the 1-node numbers behind ../amd-ubuntu/results/mi355x-vs-b200-interconnect.md §2.3
# (B200 all_reduce 18-28 us and nearly flat from 8 B to 1 MB): the default setting is run three
# times, and NCCL is also run with a fixed protocol / algorithm and with NVLS (NVLink SHARP) off,
# so NCCL's own choice can be told apart from the hardware.
# Twin of ../amd-ubuntu/rccl-tests/run_rccl_lat_1node.sh (MI355X, RCCL): same tests, sizes,
# iterations, settings and order (+ NCCL_NVLS_ENABLE=0, B200 only).
#   nccl-tests 8 B..4 MB (factor 2), 1000 iterations + 100 warm-up, no data check,
#   one MPI rank per GPU, each rank bound to its GPU's NUMA node.
#
# Submit:  sbatch job-nccl-lat-1node.sh
# Results: out-nccl-lat-1node/<job>/{STATE.txt,rccl_runs.txt,nccl_lat_1node.md}, then
#          ../amd-ubuntu/rccl-tests/compare_lat_1node.py refreshes the comparison.
#SBATCH -p mit_testing
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=200GB
#SBATCH -t 60
#SBATCH -J nccl-lat-1node
#SBATCH --exclusive
#SBATCH -o out-nccl-lat-1node/%x-%J.out

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
OUT=$PWD/out-nccl-lat-1node/$SLURM_JOB_ID; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
NCCL_DIR=/orcd/data/orcd/022/benchmarks/nccl-tests
BIN=$NCCL_DIR/build-nvhpc-26.1
AMD=/orcd/data/orcd/022/benchmarks/amd-ubuntu/rccl-tests

module purge
module load nvhpc/26.1
NVHPC_HOME=/orcd/software/core/001/pkg/nvhpc/26.1/Linux_x86_64/26.1
OMPI_HOME=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi
export LD_LIBRARY_PATH=$OMPI_HOME/lib:$LD_LIBRARY_PATH
export TMPDIR=$HOME/orcd/scratch/tmp; mkdir -p "$TMPDIR"

NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
say "1-node latency run start, job $SLURM_JOB_ID on $NODE ($OUT)"
nvidia-smi --query-gpu=index,name,clocks.sm,power.draw,utilization.gpu --format=csv,noheader > "$OUT/gpus.txt"

# per-rank NUMA binding: local rank r -> GPU r (same as job-nccl-lat-2node.sh)
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

export NCCL_DEBUG=WARN
MPI_FLAGS="--mca pml ob1 --mca btl tcp,self --mca coll_ucc_enable 0 --mca coll_hcoll_enable 0 \
   --mca btl_openib_warn_no_device_params_found 0 --bind-to none"
COLLS="sendrecv all_reduce all_gather reduce_scatter broadcast alltoall"
RUNS=$OUT/rccl_runs.txt; : > "$RUNS"

gpu_use() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ' '; }
wait_idle() {  # until every GPU reads 0 %, at most 2 min
   local u i
   for i in $(seq 1 24); do
      u=$(gpu_use)
      ! tr ' ' '\n' <<<"$u" | grep -qv '^0\?$' && { echo "idle ($u)"; return 0; }
      sleep 5
   done
   echo "NOT idle after 2 min: $u"
}

# $1 tag  $2 extra env "K=V K=V"  $3 label for the runs file
nccl() {
   local d=$OUT/$1 idle xenv="-x NCCL_DEBUG -x PATH -x LD_LIBRARY_PATH" kv; mkdir -p "$d"
   for kv in $2; do xenv+=" -x $kv"; done
   idle=$(wait_idle)
   mpirun -np 8 --host "$NODE:8" $MPI_FLAGS $xenv -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,TUNING \
      "$BIND" $BIN/all_reduce_perf -b 8 -e 64K -f 8 -g 1 -n 2 -w 1 -c 1 > "$d/nccl_info.log" 2>&1
   for c in $COLLS; do
      timeout 1200 mpirun -np 8 --host "$NODE:8" $MPI_FLAGS $xenv \
         "$BIND" $BIN/${c}_perf -b 8 -e 4M -f 2 -g 1 -n ${ITERS:-1000} -w ${WARMUP:-100} -c 0 \
         > "$d/${c}_ppn8_np8.log" 2>&1
   done
   echo "$1_1n_ppn8|$NODE|8|$3|$d" >> "$RUNS"
   say "nccl $1 (${2:-default}; GPUs before: $idle): $(for c in $COLLS; do printf '%s %sus; ' $c "$(awk '/^ *8 +[0-9]+ +[a-z]/ {print $6; exit}' "$d/${c}_ppn8_np8.log")"; done)"
}
# default three times, spread over the run; fixed protocol / algorithm in between
nccl r1     ""                              "default, run 1"
nccl ll     "NCCL_PROTO=LL"                 "NCCL_PROTO=LL"
nccl ringll "NCCL_ALGO=Ring NCCL_PROTO=LL"  "NCCL_ALGO=Ring NCCL_PROTO=LL"
nccl r2     ""                              "default, run 2"
nccl treell "NCCL_ALGO=Tree NCCL_PROTO=LL"  "NCCL_ALGO=Tree NCCL_PROTO=LL"
nccl nvls0  "NCCL_NVLS_ENABLE=0"            "NCCL_NVLS_ENABLE=0"
nccl r3     ""                              "default, run 3"

grep -h "NCCL version" "$OUT"/r1/nccl_info.log | head -1 | tee -a "$STATE"
python3 "$AMD/analyze_rccl_lat.py" "$RUNS" -o "$OUT/nccl_lat_1node.md" \
   --title "NCCL latency inside one node, $NODE (B200)" --tool nccl-tests --intra NVLink > "$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $OUT/nccl_lat_1node.md"
say "latency run DONE"
python3 "$AMD/compare_lat_1node.py" >> "$OUT/analyze.log" 2>&1
say "compare_lat_1node rc=$?"
