#!/usr/bin/env bash
# MAIN RUN: Kimi-K3 on 2 x 8 B200, TP8 x PP2, matched to the MI355X baseline in
# ../amd-benchmarks/amd-cloud/results/kimi-k3-base.md, then analyzed automatically.
#
# Submit with:   sbatch job-kimi-base.sh
#
# Sequence: preflight -> server up -> concurrency sweep -> server down -> analysis.
# The analysis runs INSIDE the job, so results/kimi-k3-base-b200.md exists the moment
# the allocation ends with no separate manual step.
#
# Why 2 nodes: the 1561 GB MXFP4 checkpoint does not fit in one node's 8 x 192 GB
# = 1538 GB of HBM, before a single byte of KV cache. TP8 shards within each node,
# PP2 splits the 93 layers across the pair. This is the vLLM recipe's verified B200
# layout (`multi_node_tp_pp`), not a tuning preference. Run job-kimi-1node.sh first if
# you want that established by measurement rather than by arithmetic.
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
#SBATCH -w node5700-c1,node5701-c1
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=b200:8
#SBATCH --mem=0
#SBATCH -t 06:00:00
#SBATCH -J kimi-k3-b200
#SBATCH --exclusive
#SBATCH -o out/kimi-k3-b200.%J.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source common/env.sh
source lib/kimi-run.sh
module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null

# Parallelism, per ./notes: "two nodes with 8 * B200 on each, with 8 * TP on each
# node and 2 * PP across two nodes".
#   TP=8  tensor-parallel WITHIN each node, over its 8 GPUs and its NVLink domain
#   PP=2  pipeline across the two nodes, one stage per node, over InfiniBand
#   world = TP x PP = 16 GPUs on 2 nodes
#
# The mapping is deliberate, not incidental: TP all-reduces twice per layer and must
# stay on NVLink, while PP crosses a stage boundary only once per step and is the only
# thing that should ever touch the inter-node fabric. Swapping them (TP across nodes)
# would put 186 all-reduces per token on InfiniBand.
export TP=8
export PP=2

kimi_run base
