#!/usr/bin/env bash
# Submit the whole benchmark as a Slurm dependency chain and return immediately.
#
# Usage: ./chain.sh [--alt]
#
# Deliberately NOT a babysitting process: no nohup, no polling loop, nothing that dies
# when the session ends. Slurm holds the chain, so it runs to completion after logout
# and would survive a login-node reboot.
#
#   pull  (CPU)      fetch the image once
#     |  afterok
#   gate  (1 node)   image + GPU + IB + arch registration
#     |  afterok
#   verify(2 nodes)  ray 16-GPU cluster + vLLM model inspection, no weight load
#     |  afterok
#   1node (1 node)   TP8 x PP1 -- EXPECTED to OOM; the failure is the measurement
#     |  afterANY on 1node  AND  afterOK on gate
#   base  (2 nodes)  TP8 x PP2 + automatic analysis -> results/kimi-k3-base-b200.md
#     |  afterany
#   summary (CPU)    -> results/RUN-SUMMARY.md, written whatever the outcome
#
# The probe stage is not in the chain: it has already run, and its verdicts are what
# selected the node set below.
set -uo pipefail
cd "$(dirname "$0")"
source common/env.sh

ALT=""
[[ "${1:-}" == "--alt" ]] && ALT="--alt"

echo "=== submitting chain ==="
echo "nodes       : $B200_NODES"
echo "reservation : ${RESV:-<none>}"
echo "model       : $MKIMI (pre-staged, read-only)"
echo "image       : $VLLM_SIF"
echo

# 1. image ------------------------------------------------------------------------
# Skip the pull entirely when the image is already present and passes check_image --
# re-queuing a no-op job just to have it exit 0 costs queue latency for nothing.
if check_image 2>/dev/null; then
  PULL=""
  echo "pull    : SKIPPED (image already present and valid)"
else
  PULL=$(sbatch --parsable job-pull-image.sh)
  echo "pull    : $PULL"
fi

# 2. gate -------------------------------------------------------------------------
if [[ -n "$PULL" ]]; then
  GATE=$(sbatch --parsable --dependency=afterok:$PULL job-gate-b200.sh)
  echo "gate    : $GATE  (afterok:$PULL)"
else
  GATE=$(sbatch --parsable job-gate-b200.sh)
  echo "gate    : $GATE  (no dependency)"
fi

# 2b. 2-node verification ----------------------------------------------------------
# Added after two failed attempts. It exercises, on real hardware and in ~10 minutes,
# the exact two steps that killed them -- ray cluster formation and vLLM model
# inspection (the Triton JIT compile) -- without paying the 1.42 TiB weight load.
# Everything downstream is afterok on THIS, so a regression in either can no longer
# reach a 6-hour 2-node allocation.
VERIFY=$(sbatch --parsable --dependency=afterok:$GATE job-verify-2node.sh)
echo "verify  : $VERIFY  (afterok:$GATE)"

# 3. single-node attempt ----------------------------------------------------------
ONE=$(sbatch --parsable --dependency=afterok:$VERIFY job-kimi-1node.sh)
echo "1node   : $ONE   (afterok:$VERIFY)"

# 4. the real run -----------------------------------------------------------------
# TWO dependencies, ANDed (Slurm treats a comma-separated list as "all must hold"):
#
#   afterok:$GATE   the gate must have PASSED. Learned the hard way: on the first
#                   attempt base depended only on afterany:$ONE, the gate failed, Slurm
#                   cancelled $ONE for an unsatisfiable dependency -- and a CANCELLED
#                   job still satisfies afterany, so base launched onto 2 B200 nodes
#                   behind a gate that had already failed.
#   afterany:$ONE   the single-node attempt is EXPECTED to fail with an OOM (the
#                   checkpoint is 23 GB larger than one node's HBM), so afterok there
#                   would cancel the rest of the chain on the predicted outcome.
BASE=$(sbatch --parsable --dependency=afterok:$VERIFY,afterany:$ONE job-kimi-base.sh)
echo "base    : $BASE  (afterok:$VERIFY AND afterany:$ONE)"

# 5. summary ----------------------------------------------------------------------
CHAIN_IDS="${PULL:+pull:$PULL }gate:$GATE verify:$VERIFY 1node:$ONE base:$BASE"
SUM=$(sbatch --parsable --dependency=afterany:$BASE \
      --export=ALL,CHAIN_IDS="$CHAIN_IDS summary:pending" job-summary.sh)
echo "summary : $SUM  (afterany:$BASE)"

{
  echo "submitted=$(date -Iseconds)"
  echo "pull=${PULL:-skipped}"
  echo "gate=$GATE"
  echo "verify=$VERIFY"
  echo "one=$ONE"
  echo "base=$BASE"
  echo "summary=$SUM"
  echo "nodes=$B200_NODES"
} >"$LOG_ROOT/CHAIN.txt"

echo
echo "chain recorded in $LOG_ROOT/CHAIN.txt"
echo
echo "Check on it later with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${PULL:+$PULL,}$GATE,$VERIFY,$ONE,$BASE,$SUM -o JobID,JobName%18,State,Elapsed,ExitCode"
echo "  cat $RESULTS/RUN-SUMMARY.md"
echo "  cat $RESULTS/kimi-k3-base-b200.md"
echo
echo "Cancel everything with:  scancel ${PULL:+$PULL }$GATE $VERIFY $ONE $BASE $SUM"
