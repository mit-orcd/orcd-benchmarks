#!/bin/bash
# Raw InfiniBand bandwidth (ib_write_bw) per rail, host-memory vs GPU-memory,
# between two nodes — used to tell a fabric-level problem from a GPUDirect one.
#
# Motivation: NCCL 2-node ring collectives run at ~half bandwidth on the routes
# node5602-c1 <-> node5802-c1 and node5702-c1 <-> node5802-c1, while the same
# node5802-c1 is at full speed with node5800-c1 (see out-nccl-2node/summary.md).
# NCCL drives all 8 rails at once, so a per-rail scan shows whether one rail (or
# one route) is slow, and running each rail twice — once host mem -> host mem,
# once GPU -> GPU — shows whether the loss is in the fabric itself (CPU path
# affected too) or only on the GPUDirect RDMA path.
#
# Submit with (node pair chosen on the command line):
#     sbatch -w nodeA,nodeB job-ibwrite-rails.sh
#
# Each rail is PXB-adjacent to exactly one GPU on this platform (nvidia-smi
# topo -m): mlx5_4->GPU0, 7->1, 8->2, 9->3, 10->4, 13->5, 14->6, 15->7. Each rail
# is NDR 400 Gb/s, so a healthy result is ~390-400 Gb/s on both rows.
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=80GB
#SBATCH -t 30
#SBATCH -J ibwrite-rails
#SBATCH --exclusive
#SBATCH -o out-ibwrite/%x-%J.out

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
OUT=out-ibwrite/rails-$SLURM_JOB_ID
mkdir -p "$OUT"

RAILS=(mlx5_4 mlx5_7 mlx5_8 mlx5_9 mlx5_10 mlx5_13 mlx5_14 mlx5_15)
GPUS=(0       1      2      3      4       5       6       7)
SIZE=$((64*1024*1024))
ITERS=200

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SRV=${NODES[0]}
CLI=${NODES[1]}

echo "=============================================================="
echo "ib_write_bw per-rail scan — host mem vs GPU mem"
echo "  date    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  job     : $SLURM_JOB_ID"
echo "  server  : $SRV"
echo "  client  : $CLI"
echo "  message : $SIZE bytes, $ITERS iterations, NDR rail max 400 Gb/s"
echo "=============================================================="

# one ib_write_bw pair; echoes the average BW in Gb/s (or FAILED)
run_one () {
   local dev="$1" port="$2" label="$3" srv_extra="$4" cli_extra="$5"

   srun --overlap -N1 -n1 -w "$SRV" \
      ib_write_bw -d "$dev" -p "$port" --report_gbits -s $SIZE -n $ITERS $srv_extra \
      > "$OUT/$label.srv" 2>&1 &
   local spid=$!
   sleep 4
   srun --overlap -N1 -n1 -w "$CLI" \
      ib_write_bw -d "$dev" -p "$port" --report_gbits -s $SIZE -n $ITERS $cli_extra "$SRV" \
      > "$OUT/$label.cli" 2>&1
   wait $spid 2>/dev/null

   awk -v s=$SIZE '$1==s {print $4}' "$OUT/$label.cli" | tail -1
}

port=19100
printf '\n%-10s %-6s %14s %14s\n' "rail" "gpu" "host->host" "GPU->GPU"
printf '%-10s %-6s %14s %14s\n' "----------" "------" "--------------" "--------------"

for i in "${!RAILS[@]}"; do
   dev=${RAILS[$i]}
   gpu=${GPUS[$i]}

   host_bw=$(run_one "$dev" $((port++)) "$dev-host" "" "")
   gpu_bw=$(run_one "$dev" $((port++)) "$dev-gpu" "--use_cuda=$gpu" "--use_cuda=$gpu")

   printf '%-10s %-6s %14s %14s\n' "$dev" "GPU$gpu" "${host_bw:-FAILED}" "${gpu_bw:-FAILED}"
done

echo
echo "raw perftest output: $OUT/"
echo "done: $(date '+%Y-%m-%d %H:%M:%S')"
