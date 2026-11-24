#!/bin/bash
# Hyperloom plan step 4, session A: Kimi-K3, ATOM without MTP, TP8, 12 h, kernel + framework agents on.
# Runs ON node6100 (start from login: ssh node6100 nohup bash .../hyperloom/run_step4.sh &).
# Hyperloom runs inside the ATOM SIF (the same image as our step 3 baseline) with its persistent overlay.
# Baseline server flags come from kimi_atom_reference.sh (= our step 3 ATOM k0 flags).
# Token: read from .env inside the container only; never printed.
# guard.sh runs next to it: a checkpoint every 30 min, and on a Claude token / usage limit message it
# checkpoints and stops everything (no auto-resume).
# Resume a stopped session (only when the user asks): RESUME=<session dir> bash run_step4.sh
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks; SW=$B/amd-software; W=$SW/hyperloom; H=$B/amd-ubuntu
SIF=$SW/sif/atom-dev-nightly_202610011450.sif; OV=$SW/cache/overlay/hyperloom-atom.img
MODEL=/scratch/Kimi-K3; HOURS=${HOURS:-12}; RESUME=${RESUME:-}
# Nov 12 rerun settings:
#  KERNEL=0        -> --no-kernel (GEAK off; it never returned a result and used most tokens)
#  FRAMEWORK=0     -> --no-framework-agent (no git checkout of ATOM, so it cannot author patches; it only used the time)
#  CONCS           -> conc-sweep user levels (default 1,8,64,128,256)
#  SPEC_CPU_MIN    -> base wall budget (min) of CPU helpers such as the research scout (was 10)
#  TMPDIR is node-local /dev/shm/shaohao-tmp (the shared-filesystem temp dir gave Errno 39)
#  ~/.local/bin is put on PATH so Hyperloom finds `claude` and runs helpers with --add-dir
#  (without it, helpers ran in the SDK mode that refused reads outside the working dir)
KERNEL=${KERNEL:-1}; FRAMEWORK=${FRAMEWORK:-1}; CONCS=${CONCS:-1,8,64,128,256}; SPEC_CPU_MIN=${SPEC_CPU_MIN:-20}
# Nov 17 knobs (plan-hyperloom.md §11): CONC = target user level (64); REF = reference script
#  (kimi_atom_reference.sh); FRAMEWORK_PATH = framework checkout the framework agent may patch (none)
TARGET=${TARGET:-10}; EXTRA_OPT=${EXTRA_OPT:-}   # Nov 18: --target-gain, extra optimize flags (e.g. --no-explore)
CONC=${CONC:-64}; REF=${REF:-$H/hyperloom/kimi_atom_reference.sh}; FRAMEWORK_PATH=${FRAMEWORK_PATH:-}
LTMP=/dev/shm/shaohao-tmp
L=$H/logs/hyperloom; mkdir -p "$L" "$SW/cache/tmp" /dev/shm/shaohao-ray "$LTMP"
ST=$L/step4-$(hostname -s).state
say() { echo "[$(date -Iseconds)] $*" | tee -a "$ST"; }

exec 9>/dev/shm/shaohao-gpu.lock
flock -n 9 || { say "GPU lock busy, exit"; exit 1; }
say "step4 A start: model=$MODEL hours=$HOURS framework=atom sif=$(basename "$(readlink -f "$SIF")") resume=${RESUME:-no}"
if [[ -n $RESUME ]]; then
  [[ -f $RESUME/state.json ]] || { say "RESUME dir has no state.json: $RESUME"; exit 2; }
  rm -f "$RESUME/STOPPED_TOKEN_LIMIT"
fi
if [[ -n $RESUME ]]; then RUNLOG=$L/step4-A-run-resume-$(date -u +%Y%m%dT%H%M%SZ).log
else RUNLOG=$L/step4-A-run-new-$(date -u +%Y%m%dT%H%M%SZ).log; fi   # never overwrite session A's step4-A-run.log
say "settings: conc=$CONC ref=$(basename "$REF") framework_path=${FRAMEWORK_PATH:-none} kernel=$KERNEL framework=$FRAMEWORK target=$TARGET extra_opt=${EXTRA_OPT:-none} concs=$CONCS spec_cpu_min=$SPEC_CPU_MIN tmpdir=$LTMP log=$RUNLOG"
SESSION_DIR="$RESUME" bash "$H/hyperloom/guard.sh" "$RUNLOG" "$ST" 9>&- &
GUARD=$!

apptainer exec --cleanenv --overlay "$OV" -B /orcd/data/orcd/022 -B /scratch/Kimi-K3 --pwd "$W" \
  --env TMPDIR="$LTMP" --env REPO_ROOT="$W" --env MODEL="$MODEL" --env HOURS="$HOURS" \
  --env KERNEL="$KERNEL" --env FRAMEWORK="$FRAMEWORK" --env CONCS="$CONCS" --env HYPERLOOM_SPECIALIST_CPU_BASE_MIN="$SPEC_CPU_MIN" \
  --env RAY_TMPDIR=/dev/shm/shaohao-ray --env HYPERLOOM_IMAGE=rocm/atom-dev:nightly_202610011450 \
  --env HF_HOME="$SW/cache/hf" --env XDG_CACHE_HOME="$SW/cache/xdg" --env REF="$REF" --env RESUME="$RESUME" --env CONC="$CONC" --env FRAMEWORK_PATH="$FRAMEWORK_PATH" --env TARGET="$TARGET" --env EXTRA_OPT="$EXTRA_OPT" \
  "$SIF" bash -c '
    set -a; . "$REPO_ROOT/.env"; set +a
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    ulimit -Sn 65536 2>/dev/null || true
    echo "== flash-linear-attention (Kimi-K3 needs it in this image)"
    python3 -c "import fla" 2>/dev/null || pip install --no-cache-dir flash-linear-attention 2>&1 | tail -3
    echo "== full runtime install (incl. kernel-agent part)"
    bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"; echo "install rc=$?"
    . "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    export PATH="$HOME/.local/bin:$PATH"
    echo "== claude CLI for helpers: $(command -v claude || echo MISSING)"
    NOK=""; [[ $KERNEL == 0 ]] && NOK="--no-kernel"; [[ $FRAMEWORK == 0 ]] && NOK="$NOK --no-framework-agent"
    echo "== optimize"
    python3 -m hyperloom.inference_optimizer.cli optimize \
      --framework atom --model "$MODEL" --gpu-type mi355x --tp 8 --model-class moe_mla \
      --conc "$CONC" --isl 1024 --osl 1024 --precision bf16 --max-model-len 16384 \
      --reference-script "$REF" ${FRAMEWORK_PATH:+--framework-path "$FRAMEWORK_PATH"} \
      --enable-conc-sweep --conc-sweep-concs "$CONCS" $NOK \
      --target-gain "$TARGET" $EXTRA_OPT --max-hours "$HOURS" ${RESUME:+--resume-from "$RESUME"}
    echo "optimize rc=$?"
    ray stop --force >/dev/null 2>&1 || true
  ' > "$RUNLOG" 2>&1
rc=$?
wait $GUARD; grc=$?
rm -rf "${LTMP:?}"/* 2>/dev/null   # node-local temp files of this run (RAM)
say "step4 A done rc=$rc guard rc=$grc (3 = stopped at a Claude limit) (log $RUNLOG)"
