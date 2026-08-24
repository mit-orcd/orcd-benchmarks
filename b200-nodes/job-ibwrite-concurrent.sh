#!/bin/bash
# Aggregate InfiniBand bandwidth with ALL 8 rails driven at the same time.
#
# Follow-up to job-ibwrite-rails.sh: that scan drives one rail at a time and
# every rail reaches line rate on every route, including the routes where NCCL
# ring collectives run at half speed. A single-rail test only ever loads one
# path through the fabric, so it cannot see contention between rails sharing an
# inter-switch link. This script starts all 8 ib_write_bw streams concurrently
# and reports the aggregate, which is what NCCL's multi-rail rings actually ask
# of the fabric.
#
# Healthy aggregate: 8 x ~395 Gb/s = ~3160 Gb/s (GPU mode).
#
# Submit with:
#     sbatch -w nodeA,nodeB job-ibwrite-concurrent.sh [host|gpu]   (default gpu)
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=80GB
#SBATCH -t 20
#SBATCH -J ibwrite-concurrent
#SBATCH --exclusive
#SBATCH -o out-ibwrite/%x-%J.out

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
OUT=out-ibwrite/concurrent-$SLURM_JOB_ID
mkdir -p "$OUT"

MODE="${1:-gpu}"          # gpu = GPUDirect on both ends, host = host memory
RAILS=(mlx5_4 mlx5_7 mlx5_8 mlx5_9 mlx5_10 mlx5_13 mlx5_14 mlx5_15)
GPUS=(0       1      2      3      4       5       6       7)
SIZE=$((64*1024*1024))
ITERS=500

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SRV=${NODES[0]}
CLI=${NODES[1]}

echo "=============================================================="
echo "ib_write_bw — all 8 rails concurrently"
echo "  date    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  job     : $SLURM_JOB_ID"
echo "  server  : $SRV"
echo "  client  : $CLI"
echo "  mode    : $MODE"
echo "  message : $SIZE bytes, $ITERS iterations"
echo "=============================================================="

port=19300
# start every server first, then every client, so all 8 streams overlap in time
for i in "${!RAILS[@]}"; do
   dev=${RAILS[$i]}; gpu=${GPUS[$i]}
   extra=""; [ "$MODE" = "gpu" ] && extra="--use_cuda=$gpu"
   srun --overlap -N1 -n1 -c1 -w "$SRV" \
      ib_write_bw -d "$dev" -p $((port+i)) --report_gbits -s $SIZE -n $ITERS $extra \
      > "$OUT/$dev.srv" 2>&1 &
done
sleep 6

for i in "${!RAILS[@]}"; do
   dev=${RAILS[$i]}; gpu=${GPUS[$i]}
   extra=""; [ "$MODE" = "gpu" ] && extra="--use_cuda=$gpu"
   srun --overlap -N1 -n1 -c1 -w "$CLI" \
      ib_write_bw -d "$dev" -p $((port+i)) --report_gbits -s $SIZE -n $ITERS $extra "$SRV" \
      > "$OUT/$dev.cli" 2>&1 &
done
wait

printf '\n%-10s %14s\n' "rail" "Gb/s"
printf '%-10s %14s\n' "----------" "--------------"
total=0
for dev in "${RAILS[@]}"; do
   bw=$(awk -v s=$SIZE '$1==s {print $4}' "$OUT/$dev.cli" | tail -1)
   printf '%-10s %14s\n' "$dev" "${bw:-FAILED}"
   [ -n "${bw:-}" ] && total=$(awk -v t="$total" -v b="$bw" 'BEGIN{print t+b}')
done
printf '%-10s %14s\n' "AGGREGATE" "$total"
echo
echo "raw perftest output: $OUT/"
echo "done: $(date '+%Y-%m-%d %H:%M:%S')"
