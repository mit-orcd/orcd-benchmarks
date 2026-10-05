#!/usr/bin/env bash
# Kimi-K3 with AMD's latest recipe: vLLM on MI355X, recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x
# (2026-09-25). One server per concurrency point, because the recipe tunes the server per point:
#
#   C        draft K (DSpark)  max-num-seqs  batched tok  DCP  CPU KV offload  cudagraph mode
#   1        7                 2             16384        1    -               FULL_AND_PIECEWISE
#   4        5                 8             8192         1    -               FULL_AND_PIECEWISE
#   8, 10    4                 16, 20        8192         1    Simple          FULL
#   12, 14   3                 24, 28        8192         1    Simple          FULL
#   44+      0 (spec off)      2*C           8192         8    Simple          FULL
#   cudagraph_capture_sizes = 2..MAX_SEQS*(1+K)   (the recipe's generator script)
# Points 64/128/256 are added beyond the recipe's matrix (same rule as 44+), to line up with the
# ATOM sweeps (base c<=64, maxseqs 64/128/256).
#
# Workload is amd-cloud's, not the recipe's agentic lane, so the numbers compare with the ATOM
# runs: random ISL/OSL 1024/1024, --ignore-eos, range ratio 0.8, 10*C prompts, client
# `vllm bench serve` inside the same container.
#
#   ./launch.sh node6101 atom/run_kimi_recipe.sh                 # all points
#   ./launch.sh node6101 atom/run_kimi_recipe.sh "1 4 70"        # a subset
# Knobs: CONC, ISL, OSL, OFFLOAD_BYTES (CPU KV bytes per rank; recipe 224875000000 = 1.8 TB for 8
#        ranks, of this node's 2.2 TiB), VLLM_IMG, MKIMI (weights), DRAFT (DSpark dir), PORT.
# Output: logs/<node>/kimi-recipe/atom/kimi_recipe_<ts>/c<C>/..., results/<node>/kimi-recipe/kimi-k3-recipe.{md,csv}
set -uo pipefail
export RESULTS_SUBDIR=${RESULTS_SUBDIR:-kimi-recipe}
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"

CONC="${1:-${CONC:-1 4 8 10 12 14 44 48 70 64 128 256}}"
ISL="${ISL:-1024}"; OSL="${OSL:-1024}"
IMG="${VLLM_IMG:-vllm/vllm-openai-rocm:nightly-rocm100}"
MODEL="${MKIMI:-$KIMI_MODEL}"
DRAFT="${DRAFT:-$DSPARK_MODEL}"
OFFLOAD_BYTES="${OFFLOAD_BYTES:-224875000000}"
PORT="${PORT:-8020}"
NAME="vllm-kimi-recipe"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_recipe_$TS; mkdir -p "$OUT" "$RESULTS"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "Kimi-K3 vLLM recipe sweep start: image=$IMG conc='$CONC' ISL/OSL=$ISL/$OSL out=$OUT"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy."; exit 1; }
[[ -d "$MODEL" ]] || { say "ABORT: model dir missing: $MODEL"; exit 1; }
[[ -d "$DRAFT" ]] || say "WARNING: DSpark draft missing ($DRAFT): points with speculation will fail"
docker image inspect "$IMG" >/dev/null 2>&1 || { say "ABORT: image $IMG not present"; exit 1; }
echo "$IMG -> $(docker image inspect --format '{{.Id}}' "$IMG")" > "$RESULTS/images.txt"

# Recipe parameters for concurrency $1 -> K MAXSEQS BATCHED DCP OFFLOAD GRAPH
params() {
  local c=$1 k
  if   (( c <= 1 ));  then k=7
  elif (( c <= 4 ));  then k=5
  elif (( c <= 10 )); then k=4
  elif (( c <= 14 )); then k=3
  else                     k=0; fi
  local ms=$((2 * c)) bt=8192 dcp=1 off=0 g=FULL
  (( c <= 1 )) && bt=16384
  (( c <= 4 )) && g=FULL_AND_PIECEWISE
  (( c >= 8 )) && off=1
  (( k == 0 )) && dcp=8
  echo "$k $ms $bt $dcp $off $g"
}

# Recipe rule: every size 2..maxcap. Above 256 that is >256 graphs, and capturing them overflows
# the custom all-reduce's registered-buffer table ("Rank data buffer is overflowed by 1" at graph
# 469/511 for C=256, 2026-10-02), so above 256 capture every 8th size (vLLM's default spacing).
capture_sizes() {
  local m=$1
  if (( m <= 256 )); then seq -s, 2 "$m"; return; fi
  { seq 2 256; seq 264 8 "$m"; echo "$m"; } | sort -nu | paste -sd,
}

server_cmd() {   # $1 = C, writes $2
  local c=$1 f=$2 k ms bt dcp off g maxcap
  read -r k ms bt dcp off g < <(params "$c")
  maxcap=$((ms * (1 + k)))
  {
    echo "#!/bin/bash"
    echo "# recipe point C=$c: K=$k max-num-seqs=$ms batched=$bt dcp=$dcp offload=$off graph=$g"
    echo "export VLLM_ROCM_USE_AITER=1 SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER_MOE_SITUV2=1"
    echo "export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm VLLM_USE_DIRECT_DCP_A2A=0"
    echo "export VLLM_USE_DIRECT_DCP_Q_GATHER=0 VLLM_USE_DIRECT_DCP_KV_GATHER=0"
    echo "export AITER_QUICK_REDUCE_QUANTIZATION=INT4 VLLM_ENGINE_READY_TIMEOUT_S=7200"
    echo "export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200"
    if (( k > 0 )); then echo "export VLLM_USE_BREAKABLE_CUDAGRAPH=1"
    else echo "export VLLM_USE_BREAKABLE_CUDAGRAPH=0 PYTHONHASHSEED=42"; fi
    echo "exec vllm serve /model --served-model-name moonshotai/Kimi-K3 --port $PORT \\"
    echo "  --trust-remote-code --moe-backend auto --tensor-parallel-size 8 \\"
    echo "  --load-format fastsafetensors --gpu-memory-utilization 0.9 --language-model-only \\"
    echo "  --enable-auto-tool-choice --tool-call-parser kimi_k3 --reasoning-parser kimi_k3 \\"
    echo "  --max-model-len 1048576 --enable-prefix-caching --kv-cache-dtype fp8 \\"
    echo "  --attention-config '{\"mla_prefill_backend\":\"ROCM_AITER_FA\"}' \\"
    echo "  --max-num-seqs $ms --max-num-batched-tokens $bt \\"
    # draft_load_config: fastsafetensors deadlocks on the 1-shard DSpark draft over 8 ranks (rank 0
    # never reaches its broadcast; watchdog after 600 s, 2026-10-02), so the draft loads with plain
    # safetensors; the 1.5 TB target keeps the recipe's fastsafetensors.
    if (( k > 0 )); then
      echo "  --speculative-config '{\"model\":\"/draft\",\"num_speculative_tokens\":$k,\"method\":\"dspark\",\"attention_backend\":\"ROCM_AITER_MLA\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"draft_load_config\":{\"load_format\":\"safetensors\"}}' \\"
    fi
    if (( dcp > 1 )); then
      echo "  --decode-context-parallel-size $dcp --dcp-comm-backend a2a --attention-backend ROCM_AITER_MLA \\"
    fi
    if (( off )); then
      echo "  --kv-transfer-config '{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$OFFLOAD_BYTES,\"lazy_offload\":false}}' \\"
    fi
    echo "  --compilation-config '{\"mode\":3,\"cudagraph_mode\":\"$g\",\"max_cudagraph_capture_size\":$maxcap,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$(capture_sizes "$maxcap")]}' \\"
    echo "  2>&1 | tee /out/server.log"
  } > "$f"
}

stop_server() { docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1; sleep 30; }

ok=0
for C in $CONC; do
  P=$OUT/c$C; mkdir -p "$P"
  server_cmd "$C" "$P/server_cmd.sh"
  say "C=$C: $(sed -n 2p "$P/server_cmd.sh" | cut -c3-)"
  docker run -d --name "$NAME" $(dgpu_args) \
    -v "$MODEL":/model:ro -v "$DRAFT":/draft:ro -v "$P":/out \
    -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=WARN \
    "$IMG" bash /out/server_cmd.sh >"$P/container_id.txt" 2>&1 || {
      say "C=$C: docker run failed: $(tail -3 "$P/container_id.txt")"; continue; }
  ready=0
  for i in $(seq 1 3600); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    if grep -qE "Engine core initialization failed|EngineCore failed to start" "$P/server.log" 2>/dev/null; then
      say "C=$C: engine init failed. Last 30 lines:"; tail -30 "$P/server.log" | tee -a "$STATE"; break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      say "C=$C: server exited during load. Last 30 lines:"; docker logs "$NAME" 2>&1 | tail -30 | tee -a "$STATE"; break
    fi
    (( i % 300 == 0 )) && say "  C=$C still loading (${i}s)"
    sleep 1
  done
  if (( ! ready )); then say "C=$C: server not ready, skipping"; stop_server; continue; fi
  say "C=$C: server READY, benchmarking ($((C * 10)) prompts)"
  # One untimed warm-up pass at the same concurrency compiles/fills caches before the measured run.
  timeout 3600 docker exec "$NAME" vllm bench serve --backend vllm --base-url "http://localhost:$PORT" \
      --model moonshotai/Kimi-K3 --tokenizer /model --trust-remote-code \
      --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 0.8 \
      --ignore-eos --num-prompts "$C" --max-concurrency "$C" --request-rate inf >"$P/warmup.log" 2>&1
  timeout 7200 docker exec "$NAME" vllm bench serve --backend vllm --base-url "http://localhost:$PORT" \
      --model moonshotai/Kimi-K3 --tokenizer /model --trust-remote-code \
      --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 0.8 \
      --ignore-eos --num-prompts $((C * 10)) --max-concurrency "$C" --request-rate inf \
      --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /out --result-filename "c$C.json" \
      >"$P/bench.log" 2>&1
  rc=$?
  if [[ -s $P/c$C.json ]]; then ok=$((ok + 1)); say "C=$C: done rc=$rc"
  else say "C=$C: no result json rc=$rc: $(tail -3 "$P/bench.log")"; fi
  stop_server
done

say "analysis ($ok points with results)"
# All sweeps in this log dir: a rerun of failed points merges with the earlier sweep.
$PY analyze_kimi_recipe.py "$LOG_ROOT"/atom/kimi_recipe_* -o "$RESULTS" >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> $RESULTS/kimi-k3-recipe.{md,csv}"
$PY "$BENCH_ROOT/report.py" >> "$STATE" 2>&1 || say "report.py failed"
echo "RECIPE_STATUS=$([[ $ok -gt 0 ]] && echo ok || echo failed)" >> "$STATE"
say "RECIPE SWEEP DONE"
