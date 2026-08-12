#!/bin/bash
# GPUDirect RDMA bandwidth (ib_write_bw) between two Rocky 8 B200 nodes.
#
# Re-measures the three-row table in
#   ../b200-ubuntu/out-nccl-2node/summary.md  section 4.1
# whose Rocky 8 column dates from 2026-07-13 and is flagged there as possibly
# stale. Same procedure as notes.md ("How to reproduce the perftest checks"):
# mlx5_4 + GPU0 (PXB pair on this platform), 64 MiB RDMA writes, 200 iterations.
#
#   host mem -> host mem      neither side --use_cuda   (fabric baseline)
#   NIC reads from GPU        client --use_cuda         (the capped path in 2026-07)
#   NIC writes into GPU       server --use_cuda
#   GPU -> GPU                both  --use_cuda          (closest to what NCCL does)
#
# Plus a small-message sweep, which separates per-operation cost from bandwidth:
# summary.md section 4.3 names it as the test that would decide between the
# remaining candidates. Run the same sweep on the Ubuntu nodes to compare.
#
# Submit with:
#     sbatch job-ibwrite-2node.sh
#
# Node selection: any two of node5500 / node5501 / node5502 that Slurm can give
# us in mit_testing. The --exclude list removes the non-B200 members of that
# partition. node5502 is not in mit_testing while it is down.
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --exclude=node1700,node1701,node5502x[00-07]
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=80GB
#SBATCH -t 20
#SBATCH -J ibwrite-2node
#SBATCH -o out-ibwrite/%x-%J.out

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
OUT=out-ibwrite/raw-$SLURM_JOB_ID
mkdir -p "$OUT"

DEV=mlx5_4          # NDR rail that is PXB-adjacent to GPU0 on this platform
GPU=0
SIZE=$((64*1024*1024))
ITERS=200

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SRV=${NODES[0]}
CLI=${NODES[1]}

echo "=============================================================="
echo "ib_write_bw GPUDirect RDMA — Rocky 8 B200"
echo "  date        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  job         : $SLURM_JOB_ID"
echo "  server      : $SRV"
echo "  client      : $CLI"
echo "  device      : $DEV   GPU: $GPU"
echo "  message     : $SIZE bytes, $ITERS iterations"
echo "=============================================================="

# ---------------------------------------------------------------- config dump
# Several rows of the comparison table in summary.md section 4.3 are marked
# "not verifiable from here"; this settles them from inside the job.
for n in "$SRV" "$CLI"; do
  echo
  echo "---------- system configuration: $n ----------"
  srun --overlap -N1 -n1 -w "$n" bash -c '
    echo "hostname     : $(hostname)"
    echo "kernel       : $(uname -r)"
    echo "cmdline      : $(cat /proc/cmdline)"
    echo "iommu groups : $(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)  (0 => IOMMU off)"
    echo "governor     : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    echo "cpu          : $(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2- | sed "s/^ //")"
    echo "driver       : $(sed -n "1p" /proc/driver/nvidia/version 2>/dev/null)"
    echo "mofed        : $(ofed_info -s 2>/dev/null)"
    echo "hca fw       : $(ibv_devinfo -d '"$DEV"' 2>/dev/null | grep -m1 fw_ver | awk "{print \$2}")"
    echo "peermem      : $(lsmod | grep -c nvidia_peermem) module(s)"
    echo "gpu0 <-> nic4: $(nvidia-smi topo -m 2>/dev/null | awk "/^GPU0/{print \$14}")"
  '
done

# ------------------------------------------------------------------- the tests
run_test () {
  local label="$1" srv_args="$2" cli_args="$3" port="$4" extra="${5:-}"

  srun --overlap -N1 -n1 -w "$SRV" \
    ib_write_bw -d $DEV -p $port --report_gbits -s $SIZE -n $ITERS $extra $srv_args \
    > "$OUT/$label.srv" 2>&1 &
  local spid=$!
  sleep 5
  srun --overlap -N1 -n1 -w "$CLI" \
    ib_write_bw -d $DEV -p $port --report_gbits -s $SIZE -n $ITERS $extra $cli_args "$SRV" \
    > "$OUT/$label.cli" 2>&1
  wait $spid 2>/dev/null

  local bw
  bw=$(awk -v s=$SIZE '$1==s {print $4}' "$OUT/$label.cli" | tail -1)
  printf '%-28s %s Gb/s\n' "$label" "${bw:-FAILED — see $OUT/$label.cli}"
}

echo
echo "---------- 64 MiB RDMA write ----------"
run_test "host mem -> host mem"   ""                ""                18515
run_test "NIC reads from GPU"     ""                "--use_cuda=$GPU" 18516
run_test "NIC writes into GPU"    "--use_cuda=$GPU" ""                18517
run_test "GPU -> GPU"             "--use_cuda=$GPU" "--use_cuda=$GPU" 18518

# ------------------------------------------------- small-message sweep (per-op)
echo
echo "---------- size sweep, NIC reads from GPU ----------"
srun --overlap -N1 -n1 -w "$SRV" \
  ib_write_bw -d $DEV -p 18520 --report_gbits -a -n 1000 \
  > "$OUT/sweep.srv" 2>&1 &
sweep_pid=$!
sleep 5
srun --overlap -N1 -n1 -w "$CLI" \
  ib_write_bw -d $DEV -p 18520 --report_gbits -a -n 1000 --use_cuda=$GPU "$SRV" \
  > "$OUT/sweep.cli" 2>&1
wait $sweep_pid 2>/dev/null
awk '/^ *[0-9]+ +[0-9]+ +[0-9.]+/ {printf "  %-12s %10s Gb/s  %10s Mpps\n", $1, $4, $5}' "$OUT/sweep.cli"

echo
echo "raw perftest output: $OUT/"
echo "done: $(date '+%Y-%m-%d %H:%M:%S')"
