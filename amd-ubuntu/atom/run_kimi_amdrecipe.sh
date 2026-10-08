#!/usr/bin/env bash
# Kimi-K3 with the recipe AMD sent as amd-kimi-k3-recipe.pdf ("VLLM: Kimi-K3 MXFP4 Benchmark Guide
# for AMD Instinct MI355X"): vllm/vllm-openai-rocm:v0.29.0, ONE server for the whole sweep, TP8,
# max-num-seqs 128, max-num-batched-tokens 4096, cudagraph FULL_DECODE_ONLY, no speculation.
#
# Two arms (one per node, run concurrently):
#   isl128k  AMD's own workload: ISL/OSL 128000/1000, range ratio 0.2, C = 1..128, 10*C prompts
#            -> reproduces the PDF's table (874-1235 total tok/s/GPU).
#   isl1k    same server, our workload: ISL/OSL 1024/1024, range ratio 0.8, C = 1..256, 10*C prompts
#            -> compares with the current vLLM recipe 1 (atom/run_kimi_recipe.sh) and ATOM.
#
#   ./launch.sh node6100 atom/run_kimi_amdrecipe.sh isl128k
#   ./launch.sh node6101 atom/run_kimi_amdrecipe.sh isl1k
#
# Deviations from the PDF, all forced by this site (none changes the server's tuning):
#   * apptainer via the docker shim (no --privileged, /dev/mem, --shm-size: host IPC is used).
#   * weights from node-local /scratch/Kimi-K3 (never downloaded), HF_HUB_OFFLINE=1, so the
#     client gets --tokenizer /model.
#   * VLLM_ENGINE_CORE_INIT_TIMEOUT 2400 -> 7200 (1.5 TB load + graph capture; time-out only).
#   * one short untimed warm-up before the first point.
#   * Before starting, the script waits until no other user's process is on the GPUs (the nodes
#     are not in Slurm), then takes this node's GPU lock (one of our jobs per node at a time).
# Output: logs/<node>/kimi-amd-recipe/atom/kimi_amdrecipe_<arm>_<ts>/,
#         results/<node>/kimi-amd-recipe/<arm>.{csv,md}, results/ubuntu/kimi-amd-recipe.md
set -uo pipefail
ARM=${1:?usage: $0 isl128k|isl1k}
export RESULTS_SUBDIR=kimi-amd-recipe
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$BENCH_ROOT/atom"

case $ARM in
  isl128k) ISL=128000; OSL=1000; RR=0.2; DEF_CONC="1 2 4 8 16 32 64 128"; BT=36000 ;;
  isl1k)   ISL=1024;   OSL=1024; RR=0.8; DEF_CONC="1 2 4 8 16 32 64 128 256"; BT=7200 ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
CONC="${CONC:-$DEF_CONC}"
IMG="${VLLM_IMG:-vllm/vllm-openai-rocm:v0.29.0}"
MODEL="${MKIMI:-$KIMI_MODEL}"
PORT="${PORT:-8030}"
NAME="vllm-kimi-amdrecipe"
ME=$(id -un)

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_amdrecipe_${ARM}_$TS; mkdir -p "$OUT" "$RESULTS"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $HOSTNAME_S $*" | tee -a "$STATE"; }
say "AMD-PDF Kimi-K3 recipe, arm=$ARM ISL/OSL=$ISL/$OSL rr=$RR conc='$CONC' image=$IMG out=$OUT"

# ---- 1. image (shared SIF dir; the first node builds it, the other waits on the lock) -------
SIF=$SIF_DIR/vllm-openai-rocm-${IMG#vllm/vllm-openai-rocm:}.sif
exec 8>"$SIF_DIR/.build-vllm-v0.29.0.lock"
flock 8
if [[ ! -s $SIF ]]; then
  say "building $SIF (apptainer build on this node)"
  bash /orcd/data/orcd/022/benchmarks/amd-software/setup/build-sifs-on-nodes.sh vllm-v0.29.0
  say "build rc=$? ($(tail -1 /orcd/data/orcd/022/benchmarks/amd-software/setup/logs/sif-vllm-v0.29.0.$HOSTNAME_S.out))"
fi
flock -u 8; exec 8>&-
docker image inspect "$IMG" >/dev/null 2>&1 || { say "ABORT: image $IMG not present"; echo "STATUS=failed" >>"$STATE"; exit 1; }
cp -f "$SIF.manifest" "$RESULTS/images.txt" 2>/dev/null

# ---- 2. hold while other users use the GPUs ------------------------------------------------
# Busy = a KFD (GPU) process owned by someone else, or GPU utilisation with no process of ours.
others_on_gpu() {
  local pids p u mine=0 other=""
  pids=$(rocm-smi --showpids 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')
  for p in $pids; do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [[ -z $u ]] && continue
    if [[ $u == "$ME" ]]; then mine=1; else other+="$u:$p "; fi
  done
  [[ -n $other ]] && { echo "$other"; return 0; }
  if (( ! mine )); then
    local busy; busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
    (( busy > 0 )) && { echo "$busy GPU(s) busy, owner unknown"; return 0; }
  fi
  return 1
}
wait_for_others() {   # needs two clear checks 60 s apart
  local clear=0 w
  while (( clear < 2 )); do
    if w=$(others_on_gpu); then
      say "HOLD: other users on the GPUs ($w); re-check in 5 min"; clear=0; sleep 300
    else
      clear=$((clear + 1)); (( clear < 2 )) && sleep 60
    fi
  done
  say "GPUs free of other users"
}
wait_for_others

# ---- 3. our own GPU lock (one job of ours per node) ---------------------------------------
exec 9>/dev/shm/shaohao-gpu.lock
say "waiting for the node GPU lock"; flock 9; say "got the node GPU lock"
wait_for_others          # re-check: someone may have started while we waited for the lock
[[ -d "$MODEL" ]] || { say "ABORT: model dir missing: $MODEL"; echo "STATUS=failed" >>"$STATE"; exit 1; }

# ---- 4. server: the PDF's start_server.sh ---------------------------------------------------
cat > "$OUT/server_cmd.sh" <<EOF
#!/bin/bash
export HF_HUB_OFFLINE=1 VLLM_ROCM_USE_AITER=1 SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1 AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_RUST_FRONTEND=0 VLLM_ENGINE_CORE_INIT_TIMEOUT=7200
exec vllm serve /model \\
  --served-model-name moonshotai/Kimi-K3 --host 0.0.0.0 --port $PORT \\
  --trust-remote-code --tensor-parallel-size 8 --load-format auto \\
  --gpu-memory-utilization 0.95 --mm-encoder-tp-mode data \\
  --max-num-seqs 128 --max-num-batched-tokens 4096 \\
  --compilation-config.cudagraph_mode FULL_DECODE_ONLY \\
  --compilation-config.custom_ops+ +fused_rms_norm_gated \\
  --reasoning-parser kimi_k3 \\
  2>&1 | tee /out/server.log
EOF
stop_server() { docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1; sleep 30; }
stop_server
docker run -d --name "$NAME" $(dgpu_args) -v "$MODEL":/model:ro -v "$OUT":/out \
  -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=WARN "$IMG" bash /out/server_cmd.sh >"$OUT/container_id.txt" 2>&1 \
  || { say "docker run failed: $(tail -3 "$OUT/container_id.txt")"; echo "STATUS=failed" >>"$STATE"; exit 1; }
ready=0
for i in $(seq 1 7200); do
  curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
  if grep -qE "Engine core initialization failed|EngineCore failed to start" "$OUT/server.log" 2>/dev/null; then
    say "engine init failed. Last 40 lines:"; tail -40 "$OUT/server.log" | tee -a "$STATE"; break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    say "server exited during load. Last 40 lines:"; tail -40 "$OUT/server.log" 2>/dev/null | tee -a "$STATE"; break
  fi
  (( i % 600 == 0 )) && say "  still loading (${i}s)"
  sleep 1
done
if (( ! ready )); then say "server not ready, giving up"; stop_server; echo "STATUS=failed" >>"$STATE"; exit 1; fi
say "server READY after ${i}s"

bench() {   # $1 = concurrency, $2 = prompts, $3 = result file ('' = do not save), $4 = timeout
  local extra=()
  [[ -n $3 ]] && extra=(--save-result --result-dir /out --result-filename "$3")
  timeout "$4" docker exec "$NAME" vllm bench serve \
    --host localhost --port "$PORT" --endpoint /v1/completions \
    --model moonshotai/Kimi-K3 --tokenizer /model --trust-remote-code \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio "$RR" --ignore-eos \
    --max-concurrency "$1" --num-prompts "$2" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 25,50,75,90,95,99 \
    "${extra[@]}"
}
say "warm-up (4 prompts, C=4)"
bench 4 4 "" 3600 >"$OUT/warmup.log" 2>&1

# ---- 5. the PDF's run_benchmarks.sh loop ---------------------------------------------------
ok=0
for C in $CONC; do
  f="Kimi-K3-MXFP4_isl${ISL}_osl${OSL}_c${C}.json"
  say "C=$C: $((C * 10)) prompts"
  bench "$C" $((C * 10)) "$f" "$BT" >"$OUT/bench_c$C.log" 2>&1; rc=$?
  if [[ -s $OUT/$f ]]; then ok=$((ok + 1)); say "C=$C: done rc=$rc $(grep -E 'Output token throughput|Median TPOT' "$OUT/bench_c$C.log" | tr -s ' ' | paste -sd';')"
  else say "C=$C: no result rc=$rc: $(tail -3 "$OUT/bench_c$C.log")"
       docker ps --format '{{.Names}}' | grep -qx "$NAME" || { say "server died; stopping sweep"; break; }
  fi
done
stop_server
flock -u 9; exec 9>&-     # GPUs are free again; analysis does not need them

# ---- 6. analysis --------------------------------------------------------------------------
say "analysis ($ok points)"
$PY analyze_kimi_amdrecipe.py "$ARM" "$LOG_ROOT"/atom/kimi_amdrecipe_${ARM}_* -o "$RESULTS" >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $RESULTS/$ARM.{csv,md}"
flock "$BENCH_ROOT/logs/auto/report.lock" $PY compare_kimi_amdrecipe.py >>"$OUT/analyze.log" 2>&1
say "compare rc=$? -> results/ubuntu/kimi-amd-recipe.md"
echo "STATUS=$([[ $ok -gt 0 ]] && echo ok || echo failed)" >> "$STATE"
say "AMD-PDF RECIPE $ARM DONE"
