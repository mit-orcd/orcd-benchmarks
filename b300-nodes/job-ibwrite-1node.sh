#!/bin/bash
# GPUDirect RDMA bandwidth (ib_write_bw) within ONE B300 node (node5900-c1).
#
# Same test as ../b200-nodes/job-ibwrite-1node.sh: the client and server run on
# two rails of the same node, so the client's GPU -> PCIe switch -> NIC read
# path is measured. The traffic still leaves the node and returns through the IB
# switch. Because both halves of the transfer sit on one host, treat the
# host-to-host number as a sanity check rather than a fabric measurement.
#
# The NIC names on the B300 node are not known in advance, so the script picks
# them from `nvidia-smi topo -m`: for GPU0 and GPU1 it takes the first NIC that
# is PIX/PXB-adjacent (same PCIe switch). Override with env vars if needed:
#     CLI_DEV=mlx5_0 SRV_DEV=mlx5_3 sbatch job-ibwrite-1node.sh
#
# Submit with:
#     sbatch job-ibwrite-1node.sh
#SBATCH -p mit_testing
#SBATCH -w node5900-c1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --gres=gpu:b300:8
#SBATCH --mem=80GB
#SBATCH -t 20
#SBATCH -J ibwrite-1node
#SBATCH -o out-ibwrite/%x-%J.out

set -u
cd "$SLURM_SUBMIT_DIR" || exit 1
OUT=out-ibwrite/raw-$SLURM_JOB_ID
mkdir -p "$OUT"

TOPO=$(nvidia-smi topo -m 2>/dev/null)

# nic_for_gpu <gpu index>: name (e.g. mlx5_4) of the first PIX/PXB NIC of that GPU
nic_for_gpu () {
  echo "$TOPO" | awk -v g="GPU$1" '
    NR==1 { for (i=1;i<=NF;i++) col[i]=$i; next }   # header: GPU0 .. NIC0 ..
    $1==g { for (i=2;i<=NF;i++) if (($i=="PIX"||$i=="PXB") && col[i-1] ~ /^NIC/) { nic=col[i-1]; break } }
    /^ *NIC[0-9]+:/ { sub(":","",$1); name[$1]=$2 }  # legend: NIC0: mlx5_0
    END { if (nic!="") print name[nic] }'
}

CLI_GPU=${CLI_GPU:-0}
SRV_GPU=${SRV_GPU:-1}
CLI_DEV=${CLI_DEV:-$(nic_for_gpu $CLI_GPU)}
SRV_DEV=${SRV_DEV:-$(nic_for_gpu $SRV_GPU)}
SIZE=$((64*1024*1024))
ITERS=200
HOST=$(hostname)

if [ -z "$CLI_DEV" ] || [ -z "$SRV_DEV" ] || [ "$CLI_DEV" = "$SRV_DEV" ]; then
  echo "Could not pick two distinct PIX/PXB NICs for GPU$CLI_GPU/GPU$SRV_GPU (got '$CLI_DEV' '$SRV_DEV')."
  echo "Set CLI_DEV / SRV_DEV explicitly. Topology:"
  echo "$TOPO"
  exit 1
fi

echo "=============================================================="
echo "ib_write_bw GPUDirect RDMA — B300, single node"
echo "  date        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  job         : $SLURM_JOB_ID"
echo "  node        : $HOST"
echo "  client      : $CLI_DEV + GPU$CLI_GPU   (the GPU-read path under test)"
echo "  server      : $SRV_DEV + GPU$SRV_GPU"
echo "  message     : $SIZE bytes, $ITERS iterations"
echo "=============================================================="

echo
echo "---------- system configuration: $HOST ----------"
echo "os           : $(grep -m1 PRETTY_NAME /etc/os-release | cut -d= -f2)"
echo "kernel       : $(uname -r)"
echo "cmdline      : $(cat /proc/cmdline)"
echo "iommu groups : $(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)  (0 => IOMMU off)"
echo "governor     : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "cpu          : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
echo "driver       : $(sed -n '1p' /proc/driver/nvidia/version 2>/dev/null)"
echo "mofed        : $(ofed_info -s 2>/dev/null)"
echo "hca fw       : $(ibv_devinfo -d $CLI_DEV 2>/dev/null | grep -m1 fw_ver | awk '{print $2}')"
echo "link rate    : $CLI_DEV $(cat /sys/class/infiniband/$CLI_DEV/ports/1/rate 2>/dev/null), $SRV_DEV $(cat /sys/class/infiniband/$SRV_DEV/ports/1/rate 2>/dev/null)"
echo "peermem      : $(lsmod | grep -c nvidia_peermem) module(s)"
echo
echo "$TOPO"

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
