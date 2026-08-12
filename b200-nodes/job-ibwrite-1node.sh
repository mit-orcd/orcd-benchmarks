#!/bin/bash
# GPUDirect RDMA bandwidth (ib_write_bw) within ONE Rocky 8 B200 node.
#
# Why a 1-node variant: the 2-node measurement (job-ibwrite-2node.sh) is the one
# that reproduces the table in ../b200-ubuntu/out-nccl-2node/summary.md section
# 4.1, but it needs two of node5500/5501/5502 free at the same time. When only
# one is available this runs the same test across two rails of a single node —
# client on mlx5_4 + GPU0, server on mlx5_7 + GPU1, both PXB-adjacent pairs.
#
# What it does and does not measure. The suspected bottleneck is the *client's*
# GPU -> PCIe switch -> NIC read path, and that path is identical whether the
# destination is a remote node or another rail of the same node — so the
# "NIC reads from GPU" number is directly comparable to the 2-node figure.
# The traffic still leaves the node and returns through the IB switch. What
# changes is that both halves of the transfer sit on one host, so treat the
# host-to-host baseline as a sanity check rather than a fabric measurement.
#
# Submit with:
#     sbatch job-ibwrite-1node.sh [node]      # default: whatever mit_testing gives
#SBATCH -p mit_testing
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --exclude=node1700,node1701,node5502x[00-07]
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=80GB
#SBATCH -t 20
#SBATCH -J ibwrite-1node
#SBATCH -o out-ibwrite/%x-%J.out

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
OUT=out-ibwrite/raw-$SLURM_JOB_ID
mkdir -p "$OUT"

CLI_DEV=mlx5_4;  CLI_GPU=0      # PXB pair
SRV_DEV=mlx5_7;  SRV_GPU=1      # PXB pair
SIZE=$((64*1024*1024))
ITERS=200
HOST=$(hostname)

echo "=============================================================="
echo "ib_write_bw GPUDirect RDMA — Rocky 8 B200, single node"
echo "  date        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  job         : $SLURM_JOB_ID"
echo "  node        : $HOST"
echo "  client      : $CLI_DEV + GPU$CLI_GPU   (the GPU-read path under test)"
echo "  server      : $SRV_DEV + GPU$SRV_GPU"
echo "  message     : $SIZE bytes, $ITERS iterations"
echo "=============================================================="

echo
echo "---------- system configuration: $HOST ----------"
echo "kernel       : $(uname -r)"
echo "cmdline      : $(cat /proc/cmdline)"
echo "iommu groups : $(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)  (0 => IOMMU off)"
echo "governor     : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "cpu          : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
echo "driver       : $(sed -n '1p' /proc/driver/nvidia/version 2>/dev/null)"
echo "mofed        : $(ofed_info -s 2>/dev/null)"
echo "hca fw       : $(ibv_devinfo -d $CLI_DEV 2>/dev/null | grep -m1 fw_ver | awk '{print $2}')"
echo "peermem      : $(lsmod | grep -c nvidia_peermem) module(s)"
echo "gpu0 <-> nic4: $(nvidia-smi topo -m 2>/dev/null | awk '/^GPU0/{print $14}')"

run_test () {
  local label="$1" srv_args="$2" cli_args="$3" port="$4"

  ib_write_bw -d $SRV_DEV -p $port --report_gbits -s $SIZE -n $ITERS $srv_args \
    > "$OUT/$label.srv" 2>&1 &
  local spid=$!
  sleep 3
  ib_write_bw -d $CLI_DEV -p $port --report_gbits -s $SIZE -n $ITERS $cli_args "$HOST" \
    > "$OUT/$label.cli" 2>&1
  wait $spid 2>/dev/null

  local bw
  bw=$(awk -v s=$SIZE '$1==s {print $4}' "$OUT/$label.cli" | tail -1)
  printf '%-28s %s Gb/s\n' "$label" "${bw:-FAILED — see $OUT/$label.cli}"
}

echo
echo "---------- 64 MiB RDMA write ----------"
run_test "host mem -> host mem"   ""                    ""                    18525
run_test "NIC reads from GPU"     ""                    "--use_cuda=$CLI_GPU" 18526
run_test "NIC writes into GPU"    "--use_cuda=$SRV_GPU" ""                    18527
run_test "GPU -> GPU"             "--use_cuda=$SRV_GPU" "--use_cuda=$CLI_GPU" 18528

echo
echo "---------- size sweep, NIC reads from GPU ----------"
ib_write_bw -d $SRV_DEV -p 18530 --report_gbits -a -n 1000 > "$OUT/sweep.srv" 2>&1 &
sweep_pid=$!
sleep 3
ib_write_bw -d $CLI_DEV -p 18530 --report_gbits -a -n 1000 --use_cuda=$CLI_GPU "$HOST" \
  > "$OUT/sweep.cli" 2>&1
wait $sweep_pid 2>/dev/null
awk '/^ *[0-9]+ +[0-9]+ +[0-9.]+/ {printf "  %-12s %10s Gb/s  %10s Mpps\n", $1, $4, $5}' "$OUT/sweep.cli"

echo
echo "raw perftest output: $OUT/"
echo "done: $(date '+%Y-%m-%d %H:%M:%S')"
