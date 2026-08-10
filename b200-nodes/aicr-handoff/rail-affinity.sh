#!/bin/bash
# Why does ib_write_bw --use_cuda report 18.5 GB/s while NCCL sustains ~48 GB/s
# per rail on the same nodes? Prime suspect: the rail perftest auto-picked is not
# the one with PIX affinity to the allocated GPU, so GDR crosses the CPU fabric.
# This maps GPU<->NIC topology and sweeps GPU-memory RDMA across all 8 NDR rails.
# Read-only.
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH -t 25
#SBATCH -J eng-rail
#SBATCH -o eng-rail-%J.out
SIZE=8388608; ITERS=2000
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
SERVER=${nodes[0]}; CLIENT=${nodes[1]}
echo "server=$SERVER client=$CLIENT"
echo "=== allocated GPU ==="
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader
echo "=== GPU <-> NIC topology (nvidia-smi topo -m) ==="
nvidia-smi topo -m 2>&1 | head -20
echo "=== mlx5 device -> PCI BDF map ==="
for d in /sys/class/infiniband/mlx5_*; do
  n=$(basename "$d"); bdf=$(basename "$(readlink -f "$d/device")")
  rate=$(cat "$d"/ports/1/rate 2>/dev/null | awk '{print $1}')
  echo "$n  bdf=$bdf  rate=${rate}"
done
echo "=== GPU-memory RDMA on each 400 Gb/s rail (unidirectional) ==="
for NIC in mlx5_4 mlx5_7 mlx5_8 mlx5_9 mlx5_10 mlx5_13 mlx5_14 mlx5_15; do
  echo "########## $NIC ##########"
  srun --nodes=1 --ntasks=1 -w "$SERVER" ib_write_bw -d "$NIC" -s $SIZE -n $ITERS -F --report_gbits --use_cuda=0 >/tmp/.s.$$ 2>&1 &
  sleep 3
  srun --nodes=1 --ntasks=1 -w "$CLIENT" ib_write_bw -d "$NIC" -s $SIZE -n $ITERS -F --report_gbits --use_cuda=0 "$SERVER" 2>&1 | grep -E "^ *$SIZE |Failed|error" | head -2
  wait
done
