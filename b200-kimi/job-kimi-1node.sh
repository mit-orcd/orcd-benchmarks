#!/usr/bin/env bash
# STEP 1: try Kimi-K3 on ONE node, TP=8. Expected to fail on memory — and that failure
# is the measurement.
#
# Submit with:   sbatch job-kimi-1node.sh
#
# The arithmetic says it cannot work: 8 x 192 GB = 1538 GB of HBM against 1561 GB of
# weights, so a single node is ~23 GB short before a byte of KV cache, activation
# workspace, NCCL buffers or CUDA-graph pool (another ~15-20 GB/GPU). The vLLM recipe
# independently marks `single_node_tp` unusable for this model on B200.
#
# It is still worth running, for two reasons: measured beats derived, and if the nodes
# turn out to carry more usable HBM than nvidia-smi's spec figure suggests, the whole
# benchmark collapses to one node and gets much simpler. One node for well under an
# hour is a cheap way to buy that certainty.
#
# On OOM the job says so explicitly and points at job-kimi-base.sh. If it unexpectedly
# SUCCEEDS, it runs the full sweep and writes results/kimi-k3-base-b200-1node.md.
#
# Slurm targeting comes from ./notes and was verified live against `scontrol show res`:
#   reservation rres_joohye_2026-08-20_lj4j2ya3   ACTIVE 2026-08-20 -> 2026-08-27
#   Nodes=node5700-c1,node5701-c1  PartitionName=mit_testing
#   Accounts=rres_acc_joohye_2026-08-20_lj4j2ya3
# The account flag is required: the reservation is account-restricted, so a job charged
# anywhere else is refused entry to it.
# QOS is deliberately left at the default (`normal`). The reservation carries its own
# QOS `rres_qos_joohye_2026-08-20_lj4j2ya3`, but it is flagged RequiresReservation +
# OverPartQOS and exists to lift partition limits we do not hit -- mit_testing already
# allows MaxTime=7-00:00:00 and AllowQos=normal,unlimited. If Slurm ever refuses the
# submission, add: #SBATCH -q rres_qos_joohye_2026-08-20_lj4j2ya3
#SBATCH -p mit_testing
#SBATCH --reservation=rres_joohye_2026-08-20_lj4j2ya3
#SBATCH -A rres_acc_joohye_2026-08-20_lj4j2ya3
#SBATCH -w node5700-c1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=b200:8
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -J kimi-k3-1node
#SBATCH --exclusive
#SBATCH -o out/kimi-k3-1node.%J.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source common/env.sh
source lib/kimi-run.sh
module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null

# Parallelism, per ./notes: "8 * B200 on one node, with 8 * TP".
#   TP=8  tensor-parallel across all 8 GPUs of node5700-c1
#   PP=1  no pipeline stage -- everything is on one node
#   world = TP x PP = 8 GPUs
export TP=8
export PP=1
# 0.95 rather than the 2-node profile's 0.90: with no pipeline stage there is no
# per-stage prefill headroom to protect, and if this configuration has any chance at
# all it needs every byte. (If it OOMs at 0.95 it would certainly OOM at 0.90.)
export GPU_MEM_UTIL="${GPU_MEM_UTIL_1NODE:-0.95}"
# Shorter than the 2-node timeout: an OOM surfaces during weight load, so there is no
# reason to hold 8 GPUs for a full hour waiting for a load that is going to die.
export READY_TIMEOUT="${READY_TIMEOUT_1NODE:-2400}"

kimi_run 1node
