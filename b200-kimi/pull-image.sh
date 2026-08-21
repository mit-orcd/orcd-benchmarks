#!/usr/bin/env bash
# Pull the vLLM Kimi-K3 image into imag/ as an Apptainer .sif -- EXACTLY ONCE.
#
# Usage: ./pull-image.sh [--force] [--verify] [--keep-cache] [tag]
#   --force       discard an existing image and refetch (default: keep it, exit 0)
#   --verify      recompute the sha256 of the existing image and check the manifest
#   --keep-cache  keep the OCI layer cache (default: delete it once the .sif is built)
#   tag           image tag (default: $VLLM_TAG, i.e. kimi-k3)
#
# Runs on the login node -- network + disk, no GPU. It is the one step here that is not
# a Slurm job, because apptainer's docker->sif conversion is a single CPU-bound squashfs
# build and the login node is where the outbound network is. Kept polite: `nice`, a
# single attempt, and no retry storm.
#
# THE IMAGE IS PULLED HERE AND NOWHERE ELSE. Every other script calls check_image() and
# fails if it is missing. A GPU job must never spend allocation time fetching 15 GB, and
# two jobs must never race to write the same .sif.
set -uo pipefail
cd "$(dirname "$0")"
source common/env.sh

FORCE=0; VERIFY=0; KEEP_CACHE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)      FORCE=1; shift ;;
    --verify)     VERIFY=1; shift ;;
    --keep-cache) KEEP_CACHE=1; shift ;;
    -h|--help)    sed -n '2,18p' "$0"; exit 0 ;;
    *)            VLLM_TAG="$1"; shift ;;
  esac
done

SIF="$IMG_DIR/vllm-openai_${VLLM_TAG}.sif"
MAN="$SIF.manifest"
LOCK="$SIF.lock"
REF="docker://vllm/vllm-openai:${VLLM_TAG}"

module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null

# Serialize on a lock file so two shells cannot pull the same image concurrently. flock
# holds for the whole script; a second invocation waits and then finds the finished
# image and exits, rather than fighting over the destination.
exec 9>"$LOCK"
if ! flock -w 7200 9; then
  echo "ERROR: another pull-image.sh holds $LOCK and did not finish within 2 h" >&2
  exit 1
fi

# ---- already have it? ------------------------------------------------------------
if [[ $FORCE -eq 0 && -f "$SIF" && -f "$MAN" ]]; then
  if check_image "$SIF"; then
    echo "image already present, not refetching:"
    echo "  $SIF ($(du -h "$SIF" | cut -f1))"
    sed 's/^/  /' "$MAN"
    if [[ $VERIFY -eq 1 ]]; then
      echo "verifying sha256 (this reads the whole file; a few minutes)..."
      want=$(awk -F= '/^sha256=/{print $2}' "$MAN")
      have=$(sha256sum "$SIF" | cut -d' ' -f1)
      if [[ "$want" == "$have" ]]; then
        echo "  sha256 OK: $have"
      else
        echo "  MISMATCH: manifest=$want actual=$have" >&2
        echo "  Re-run with --force to discard and refetch." >&2
        exit 1
      fi
    else
      echo "(pass --verify to recompute the sha256)"
    fi
    exit 0
  fi
  echo "existing image failed its checks; refetching" >&2
fi

if [[ -f "$SIF" && $FORCE -eq 1 ]]; then
  echo "--force: removing existing $SIF"
  rm -f "$SIF" "$MAN"
fi
# A .sif with no manifest is the signature of an interrupted previous pull. Never keep
# it: it is the file that would otherwise pass a naive `-f` test and crash a GPU job.
if [[ -f "$SIF" && ! -f "$MAN" ]]; then
  echo "found $SIF with no manifest -- an interrupted previous pull. Discarding it."
  rm -f "$SIF"
fi

# ---- space ------------------------------------------------------------------------
# Both caches are forced off $HOME by common/env.sh. Re-state it here because this is
# the step that would actually blow the quota: $HOME is at ~441/500 GB, the compressed
# image is 15.4 GB, and apptainer holds the OCI layers AND the build scratch before it
# writes the final .sif.
echo "ref   : $REF"
echo "dest  : $SIF"
echo "cache : $APPTAINER_CACHEDIR"
echo "tmp   : $APPTAINER_TMPDIR"
df -h "$IMG_DIR" | tail -1

avail_gb=$(df -BG --output=avail "$IMG_DIR" | tail -1 | tr -dc '0-9')
if [[ "${avail_gb:-0}" -lt 120 ]]; then
  echo "REFUSING: only ${avail_gb} GB free; the pull needs ~100 GB transiently" >&2
  exit 1
fi

# ---- pull to a TEMP name, then rename ---------------------------------------------
# `apptainer pull` writes its destination in place, so interrupting it leaves a
# truncated file at the final path. Building under a temp name and renaming only on
# success means $SIF either does not exist or is complete -- never half-written.
TMP="$IMG_DIR/.pull-${VLLM_TAG}.$$.sif"
cleanup() { rm -f "$TMP"; }
trap cleanup EXIT INT TERM

echo "[$(date -Iseconds)] pulling -> $TMP"
t0=$(date +%s)
nice -n 10 apptainer pull "$TMP" "$REF"
rc=$?
dur=$(( $(date +%s) - t0 ))
echo "[$(date -Iseconds)] rc=$rc after ${dur}s"
if [[ $rc -ne 0 || ! -s "$TMP" ]]; then
  echo "pull FAILED -- nothing was written to $SIF. Re-run to resume from the layer cache." >&2
  exit 1
fi

# ---- verify before publishing ------------------------------------------------------
echo "inspecting the built image before publishing it"
if ! apptainer inspect "$TMP" >"$IMG_DIR/.inspect.$$" 2>&1; then
  echo "ERROR: apptainer cannot inspect the built image; not publishing it." >&2
  cat "$IMG_DIR/.inspect.$$" >&2; rm -f "$IMG_DIR/.inspect.$$"
  exit 1
fi
rm -f "$IMG_DIR/.inspect.$$"

BYTES=$(stat -c %s "$TMP")
echo "computing sha256 (once, so later checks can be a cheap size test)..."
SHA=$(sha256sum "$TMP" | cut -d' ' -f1)

mv -f "$TMP" "$SIF"
trap - EXIT INT TERM

# Manifest last: its existence is what marks the image complete, so it must not appear
# before the .sif is fully in place.
{
  echo "ref=$REF"
  echo "tag=$VLLM_TAG"
  echo "bytes=$BYTES"
  echo "sha256=$SHA"
  echo "pulled=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "apptainer=$(apptainer --version 2>/dev/null)"
  echo "pull_seconds=$dur"
} >"$MAN"

echo
ls -lh "$SIF"
sed 's/^/  /' "$MAN"

# ---- drop the layer cache ----------------------------------------------------------
# The OCI layer cache is a second full copy of the image and is only useful for
# resuming an interrupted pull. Once the .sif exists it is dead weight on a shared
# filesystem, so it goes unless asked to stay.
if [[ $KEEP_CACHE -eq 0 && -d "$APPTAINER_CACHEDIR" ]]; then
  csz=$(du -sh "$APPTAINER_CACHEDIR" 2>/dev/null | cut -f1)
  rm -rf "${APPTAINER_CACHEDIR:?}"/* 2>/dev/null
  echo "removed layer cache (${csz:-?}) -- pass --keep-cache to retain it"
fi

echo
echo "sanity: vllm version inside the image"
apptainer exec --env PYTHONNOUSERSITE=1 "$SIF" python3 -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | sed 's/^/  /' \
  || echo "  (import needs a GPU node; job-gate-b200.sh checks this properly)"
echo
echo "DONE. The image is now pulled and will not be fetched again."
echo "Every other script calls check_image() and fails if it is missing -- nothing else pulls."
