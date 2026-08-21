#!/usr/bin/env bash
# Fetch the SMOKE-TIER model only. Kimi-K3 is NOT downloaded — it is already staged on
# the cluster at /orcd/compute/orcd/025/models/Kimi-K3 and is used in place.
#
# Submit with:   sbatch download-kimi.sh [smoke]
#
# A batch job rather than a login-node command: even 8.9 GB of network + shared-NFS I/O
# belongs off the login node. No GPU is requested.
#
# `hf download` is idempotent and resumable, so an interruption is a re-submit rather
# than a restart.
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J kimi-dl
#SBATCH -o out/kimi-dl.%J.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source common/env.sh

WHICH="${1:-smoke}"

if [[ "$WHICH" == "kimi" || "$WHICH" == "all" ]]; then
  cat >&2 <<MSG
Refusing to download Kimi-K3.

It is already staged on the cluster and is used directly:
  $MKIMI
  -> $(readlink -f "$MKIMI" 2>/dev/null || echo '(not resolvable from here)')

Re-downloading 1.56 TB that already exists would waste hours of network and shared-NFS
I/O and 1.42 TiB of quota for no benefit. If you believe the staged copy is wrong, check
it first:

  source common/env.sh && check_model && echo OK

That verifies the shard count and exact byte total without reading the weights.
MSG
  exit 1
fi

# --- smoke tier only --------------------------------------------------------------
REPO="Qwen/Qwen3-8B-FP8"
DEST="$MSMOKE"
echo "[$(date -Iseconds)] smoke tier: $REPO -> $DEST (~8.9 GB)"
df -h "$MODELS" | tail -1

command -v hf >/dev/null 2>&1 || {
  # The image carries huggingface_hub; borrow it rather than requiring a host venv. It
  # is only ever consumed here -- if it has not been pulled yet, say so and stop rather
  # than pulling 15 GB from inside a batch job.
  module load "$APPTAINER_MODULE" 2>/dev/null
  check_image || {
    echo "No host \`hf\` and no image to borrow one from." >&2
    echo "Run ./pull-image.sh on the login node first, or install huggingface_hub." >&2
    exit 1
  }
  HF_CMD=(apptainer exec --bind "$BENCH_ROOT" --env HF_HOME="$HF_HOME" "$VLLM_SIF" hf)
}
: "${HF_CMD:=}"
[[ -z "${HF_CMD:-}" ]] && HF_CMD=(hf)

nice -n 10 "${HF_CMD[@]}" download "$REPO" --local-dir "$DEST" --max-workers 4
rc=$?
echo "[$(date -Iseconds)] rc=$rc size=$(du -sh "$DEST" 2>/dev/null | cut -f1)"
[[ $rc -eq 0 ]] || { echo "FAILED -- re-submit to resume" >&2; exit 1; }

echo "[$(date -Iseconds)] done"
df -h "$MODELS" | tail -1
