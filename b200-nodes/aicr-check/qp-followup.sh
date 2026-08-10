#!/bin/bash
# Follow-up: does the GPU-memory RDMA cap seen in C3/C4 depend on QP count?
# Engaging NCCL reaches ~48 GB/s per rail on GPU memory, yet single-QP
# ib_write_bw reports only 18.5 GB/s on the same rail. NCCL uses several
# QPs/channels per connection; perftest defaults to one. This sweeps -q to test
# whether that alone explains the gap. Read-only, nothing is reconfigured.
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH -t 20
#SBATCH -J eng-qp
#SBATCH -o eng-qp-%J.out
NIC=${NIC_FORCE:-mlx5_10}
SIZE=8388608; ITERS=2000
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
SERVER=${nodes[0]}; CLIENT=${nodes[1]}
echo "server=$SERVER client=$CLIENT rail=$NIC size=$SIZE"
echo "perftest: $(ib_write_bw --version 2>&1 | head -1)"
run() { # $1 label, rest = flags
  local label="$1"; shift
  echo "########## $label ##########"
  srun --nodes=1 --ntasks=1 -w "$SERVER" ib_write_bw -d "$NIC" -s $SIZE -n $ITERS -F --report_gbits "$@" >/tmp/.s.$$ 2>&1 &
  sleep 3
  srun --nodes=1 --ntasks=1 -w "$CLIENT" ib_write_bw -d "$NIC" -s $SIZE -n $ITERS -F --report_gbits "$@" "$SERVER" 2>&1 | grep -E "^ *$SIZE |Failed|error"
  wait
}
for q in 1 2 4 8 16; do
  run "GPU unidirectional, -q $q" --use_cuda=0 -q $q
done
for q in 1 2 4 8 16; do
  run "GPU bidirectional,  -q $q" --use_cuda=0 -b -q $q
done
echo "########## HOST unidirectional, -q 1 (control) ##########"
run "host uni q1" -q 1
