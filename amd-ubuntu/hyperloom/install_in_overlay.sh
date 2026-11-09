#!/bin/bash
# Hyperloom install.sh inside a SIF, saved to its own persistent overlay. Run on the LOGIN node (internet).
# --skip-kernel-agent: the kernel-agent installer starts a Ray head, so it runs on the AMD node instead.
#   install_in_overlay.sh <name> <sif basename> [install.sh options]
set -uo pipefail
NAME=${1:?overlay name}; SIFB=${2:?sif}; shift 2
SW=/orcd/data/orcd/022/benchmarks/amd-software; W=$SW/hyperloom; SIF=$SW/sif/$SIFB.sif
OV=$SW/cache/overlay/hyperloom-$NAME.img
SCR=$(realpath "$HOME/orcd/scratch"); mkdir -p "$SCR/tmp" "$SCR/pip-cache"
[[ -f $OV ]] || apptainer overlay create --sparse --size 40960 "$OV" || exit 3
nice apptainer exec --cleanenv --overlay "$OV" -B /orcd/data/orcd/022 -B "$SCR" --pwd "$W" \
  --env TMPDIR="$SCR/tmp" --env PIP_CACHE_DIR="$SCR/pip-cache" --env REPO_ROOT="$W" --env PYTHONPATH="$W" \
  "$SIF" bash -c 'set -a; . "$REPO_ROOT/.env"; set +a; export HYPERLOOM_RUNTIME_DIR="$USER_DATA_PATH/runtime-'"$NAME"'"; ulimit -Sn 65536 || true; bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh" "$@"; echo "install rc=$?"; python3 -c "import sys,yaml,claude_agent_sdk,openai,httpx; print(sys.prefix, claude_agent_sdk.__version__)"; python3 -c "import magpie; print(magpie.__file__)"' _ "$@"
