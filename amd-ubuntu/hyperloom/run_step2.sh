#!/bin/bash
# Hyperloom plan step 2: small token test. Qwen3-8B, 1 GPU, vLLM image, 1 hour, no kernel agent.
# Runs ON node6100 (start from login: ./launch.sh node6100 hyperloom/run_step2.sh).
# Hyperloom runs inside the vLLM SIF with its persistent overlay (apptainer, no docker).
# Token: read from .env inside the container only; never printed.
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks; SW=$B/amd-software; W=$SW/hyperloom
SIF=$SW/sif/vllm-openai-rocm-v0.29.0.sif; OV=$SW/cache/overlay/hyperloom-vllm.img
MODEL=$SW/models/Qwen3-8B; HOURS=${HOURS:-1}
L=$B/amd-ubuntu/logs/hyperloom; mkdir -p "$L" "$SW/cache/tmp" /dev/shm/shaohao-ray   # Ray: AF_UNIX socket path must be < 108 bytes
ST=$L/step2-$(hostname -s).state
say() { echo "[$(date -Iseconds)] $*" | tee -a "$ST"; }

exec 9>/dev/shm/shaohao-gpu.lock
flock -n 9 || { say "GPU lock busy, exit"; exit 1; }
say "step2 start: model=$MODEL hours=$HOURS framework=vllm sif=$(basename "$SIF")"

apptainer exec --cleanenv --overlay "$OV" -B /orcd/data/orcd/022 --pwd "$W" \
  --env TMPDIR="$SW/cache/tmp" --env REPO_ROOT="$W" --env MODEL="$MODEL" --env HOURS="$HOURS" \
  --env RAY_TMPDIR=/dev/shm/shaohao-ray --env HYPERLOOM_IMAGE=vllm/vllm-openai-rocm:v0.29.0 \
  --env HF_HOME="$SW/cache/hf" --env XDG_CACHE_HOME="$SW/cache/xdg" \
  "$SIF" bash -c '
    set -a; . "$REPO_ROOT/.env"; set +a
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    ulimit -Sn 65536 2>/dev/null || true
    echo "== full runtime install (incl. kernel-agent part)"
    bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"; echo "install rc=$?"
    . "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    echo "== optimize"
    python3 -m hyperloom.inference_optimizer.cli optimize \
      --framework vllm --model "$MODEL" --gpu-type mi355x --tp 1 \
      --conc 64 --isl 1024 --osl 1024 --precision bf16 \
      --target-gain 30 --max-hours "$HOURS" \
      --max-minutes-explore-pct 0.39 --max-minutes-sweep-pct 0.01 --explore-force-exit-budget-pct 0.01 \
      --no-framework-agent --no-kernel --no-enable-conc-sweep --no-enable-roofline
    echo "optimize rc=$?"
    ray stop --force >/dev/null 2>&1 || true
  ' > "$L/step2-run.log" 2>&1
say "step2 done rc=$? (log $L/step2-run.log)"
