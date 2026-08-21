#!/usr/bin/env bash
# Pull the vLLM image as a batch job, so the chain can run unattended after logout.
#
# Submit with:   sbatch job-pull-image.sh
#
# A CPU job, not a login-node command: this moves ~15 GB over the network and then
# builds a ~20 GB squashfs, which does not belong on a shared login node. Compute-node
# outbound network was verified before this was wired in (registry-1.docker.io and
# huggingface.co both reachable from mit_normal).
#
# Delegates entirely to pull-image.sh, which holds the flock + atomic-rename + manifest
# logic. If the image is already present and valid this exits 0 immediately, so the job
# is safe to leave in a dependency chain.
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J kimi-pull
#SBATCH -o out/kimi-pull.%J.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source common/env.sh

echo "[$(date -Iseconds)] pulling image on $SLURMD_NODENAME"
./pull-image.sh
rc=$?
echo "[$(date -Iseconds)] pull-image.sh rc=$rc"

# The chain depends on this job's exit status, so make the readiness test explicit
# rather than trusting the pull's own return code.
if check_image; then
  echo "IMAGE READY: $VLLM_SIF"
  sed 's/^/  /' "$VLLM_SIF_MANIFEST"
  exit 0
fi
echo "IMAGE NOT READY -- the dependent jobs will not run." >&2
exit 1
