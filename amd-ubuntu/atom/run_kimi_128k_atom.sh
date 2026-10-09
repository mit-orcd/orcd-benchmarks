#!/usr/bin/env bash
# Kimi-K3 with ATOM on AMD's own workload from amd-kimi-k3-recipe.pdf (ISL/OSL 128000/1000,
# range ratio 0.2, C = 1..128, 10*C prompts), so ATOM can be compared with AMD's published numbers
# and with vLLM recipe 1 / 2 on the same workload. Server = the ATOM base recipe (same image and
# flags as the other ATOM runs, max-num-seqs 64) with max-model-len raised to fit 128K prompts.
# If the first point fails (e.g. no chunked prefill for prompts > max-num-batched-tokens), the
# server is restarted once with max-num-batched-tokens = max-model-len.
# Run through run_kimi_128k.sh (waits for other users, takes the node GPU lock, analyzes).
# Output: logs/<node>/kimi-128k/atom/kimi_128k_atom_<ts>/Kimi-K3-MXFP4_isl128000_osl1000_c<C>.json
set -uo pipefail
export RESULTS_SUBDIR=${RESULTS_SUBDIR:-kimi-128k}
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"

IMG="${ATOM_IMG:-rocm/atom-dev:latest}"
MODEL="${MKIMI:-$KIMI_MODEL}"
PORT="${PORT:-8014}"
MAXSEQS="${MAXSEQS:-64}"
CONC="${1:-${CONC:-1 2 4 8 16 32 64 128}}"
ISL="${ISL:-128000}"; OSL="${OSL:-1000}"; RR="${RR:-0.2}"
MAXLEN="${MAXLEN:-163840}"
BT="${BT:-16384}"
NAME="atom-kimi-128k"

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_128k_atom_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "Kimi-K3 ATOM on AMD's 128K/1K workload: image=$IMG max-num-seqs=$MAXSEQS conc='$CONC' ISL/OSL=$ISL/$OSL rr=$RR out=$OUT"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy."; exit 1; }
[[ -d "$MODEL" ]] || { say "ABORT: model dir missing: $MODEL"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { say "ABORT: image $IMG not present"; exit 1; }
docker rm -f "$NAME" >/dev/null 2>&1

KIMI_QUANT='{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'

start_server() {   # $1 = max-num-batched-tokens
  local bt=$1 CMD=$OUT/server_cmd_bt$1.sh
  {
    echo '#!/usr/bin/env bash'
    echo 'set -uo pipefail'
    echo "if ! python -c 'import fla' 2>/dev/null; then pip install --no-cache-dir flash-linear-attention 2>&1 | tail -3; fi"
    echo 'exec python -m atom.entrypoints.openai_server \'
    echo '  --model /model --tensor-parallel-size 8 \'
    echo "  --server-port $PORT --kv_cache_dtype fp8 --max-num-seqs $MAXSEQS \\"
    echo '  --gpu-memory-utilization 0.93 --trust-remote-code \'
    echo "  --max-model-len $MAXLEN --max-num-batched-tokens $bt \\"
    echo '  --block-size 128 --no-enable_prefix_caching \'
    printf "  --online_quant_config '%s' 2>&1 | tee -a /out/atom_server.log\n" "$KIMI_QUANT"
  } >"$CMD"; chmod +x "$CMD"
  say "starting ATOM server (max-num-batched-tokens=$bt)"
  docker run -d --name "$NAME" $(dgpu_args) \
    -v "$MODEL":/model:ro -v "$OUT":/out \
    -e AITER_LOG_LEVEL=WARNING -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
    "$IMG" bash "/out/$(basename "$CMD")" >"$OUT/container_id.txt" 2>&1 || { say "docker run failed"; return 1; }
  for i in $(seq 1 3600); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { say "server READY"; return 0; }
    if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      say "server exited during load. Last 30 lines:"; docker logs "$NAME" 2>&1 | tail -30 | tee -a "$STATE"
      docker rm -f "$NAME" >/dev/null 2>&1; return 1
    fi
    (( i % 300 == 0 )) && say "  still loading (${i}s)"
    sleep 1
  done
  say "server not ready in 3600 s"; docker rm -f "$NAME" >/dev/null 2>&1; return 1
}
stop_server() { docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1; sleep 10; }

bench() {   # $1 = C ; returns 0 if requests completed
  local C=$1 json=Kimi-K3-MXFP4_isl${ISL}_osl${OSL}_c$1.json
  # one short untimed warm-up (min(C, 4) prompts), like vLLM recipe 2's run
  timeout 3600 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
      --model /model --backend vllm --base-url "http://localhost:${PORT}" \
      --dataset-name random --ignore-eos --request-rate inf --random-range-ratio "$RR" \
      --trust-remote-code --max-concurrency "$C" --num-prompts "$(( C < 4 ? C : 4 ))" \
      --random-input-len "$ISL" --random-output-len "$OSL" >"$OUT/warmup_c$C.log" 2>&1
  timeout 36000 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
      --model /model --backend vllm --base-url "http://localhost:${PORT}" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --dataset-name random --ignore-eos --request-rate inf --random-range-ratio "$RR" \
      --trust-remote-code --max-concurrency "$C" --num-prompts $((C * 10)) \
      --random-input-len "$ISL" --random-output-len "$OSL" --save-result \
      --result-dir /out --result-filename "$json" >"$OUT/c$C.log" 2>&1
  local rc=$? done_n=0
  [[ -s $OUT/$json ]] && done_n=$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('completed',0))" "$OUT/$json")
  say "C=$C: rc=$rc completed=$done_n"
  (( done_n > 0 ))
}

start_server "$BT" || { (( BT < MAXLEN )) && { BT=$MAXLEN; start_server "$BT"; } || { say "ABORT: server failed"; exit 1; }; }
first=1
for C in $CONC; do
  if ! bench "$C"; then
    if (( first && BT < MAXLEN )); then
      say "first point failed with max-num-batched-tokens=$BT; retrying with $MAXLEN"
      stop_server; BT=$MAXLEN
      start_server "$BT" || { say "ABORT: server failed"; exit 1; }
      bench "$C" || { say "ABORT: C=$C failed again"; stop_server; exit 1; }
    else
      say "C=$C failed; continuing"
    fi
  fi
  first=0
done
stop_server
echo "ATOM128K_SWEEP=$OUT" >>"$STATE"
say "ATOM 128K SWEEP DONE"
