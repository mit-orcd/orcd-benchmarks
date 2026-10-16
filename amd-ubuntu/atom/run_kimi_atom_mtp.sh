#!/usr/bin/env bash
# Kimi-K3 on 8 x MI355X with ATOM + DSpark speculative decoding (MTP).
# Same server and workload as our ATOM baseline (run_part_d.sh tier3: image
# rocm/atom-dev:nightly_202608111555, TP8, max-num-seqs 64, gpu-mem-util 0.93,
# max-model-len 16384, batched 16384, block 128, no prefix caching, PTPC-FP8 quant;
# ISL/OSL 1024/1024, --ignore-eos, range ratio 0.8, 10 x C prompts) plus
#   --method dspark --draft-model Inferact/Kimi-K3-DSpark --num-speculative-tokens K
# K as in the SemiAnalysis InferenceX ATOM recipe: 6 at 1 user, 2 above (one server each).
# Real acceptance: no --spec-decode-acceptance-length.
#
#   ./launch.sh node6100 atom/run_kimi_atom_mtp.sh
# Knobs: ARMS ("K:conc list;K:conc list"), ATOM_IMG, DRAFT, PORT.
# Output: logs/<node>/kimi-atom-mtp/atom/kimi_atom_mtp_<ts>/k<K>/c<C>.json; at the end
# atom/analyze_atom_mtp.py updates results/mi355x-vs-b200-kimi-k3.md, then sync and push.
set -uo pipefail
export RESULTS_SUBDIR=${RESULTS_SUBDIR:-kimi-atom-mtp}
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"

IMG="${ATOM_IMG:-${ATOM_IMG_CLOUD:-rocm/atom-dev:nightly_202608111555}}"
MODEL="${MKIMI:-$KIMI_MODEL}"
DRAFT="${DRAFT:-$DSPARK_MODEL}"
PORT="${PORT:-8031}"
ARMS="${ARMS:-6:1;2:2 4 8 10 16 32 48 64}"
ISL=1024; OSL=1024; MAXSEQS=64
NAME="atom-kimi-mtp"

exec 9>/dev/shm/shaohao-gpu.lock
flock 9   # one GPU job per node

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_atom_mtp_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "Kimi-K3 ATOM + MTP start: image=$IMG draft=$DRAFT arms='$ARMS' out=$OUT"

busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
[[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $busy GPU(s) busy"; exit 1; }
pgrep -af 'atom.entrypoints' >/dev/null 2>&1 && { say "ABORT: another ATOM server is running"; exit 1; }
[[ -d $MODEL && -s $DRAFT/model.safetensors ]] || { say "ABORT: model or draft missing ($MODEL, $DRAFT)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { say "ABORT: image $IMG not present"; exit 1; }
echo "$IMG -> $(docker image inspect --format '{{.Id}}' "$IMG")" >"$OUT/images.txt"

KIMI_QUANT='{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'

run_arm() {   # $1 = K, $2 = conc list
  local K=$1 CONC=$2 D=$OUT/k$1 ready=0 C n rc
  mkdir -p "$D"
  docker rm -f "$NAME" >/dev/null 2>&1
  {
    echo '#!/usr/bin/env bash'
    echo 'set -uo pipefail'
    echo "python -c 'import fla' 2>/dev/null || pip install --no-cache-dir flash-linear-attention 2>&1 | tail -3"
    echo 'exec python -m atom.entrypoints.openai_server \'
    echo '  --model /model --tensor-parallel-size 8 \'
    echo "  --server-port $PORT --kv_cache_dtype fp8 --max-num-seqs $MAXSEQS \\"
    echo '  --gpu-memory-utilization 0.93 --trust-remote-code \'
    echo '  --max-model-len 16384 --max-num-batched-tokens 16384 --block-size 128 \'
    echo '  --no-enable_prefix_caching \'
    echo "  --method dspark --draft-model /draft --num-speculative-tokens $K \\"
    printf "  --online_quant_config '%s' 2>&1 | tee /out/atom_server.log\n" "$KIMI_QUANT"
  } >"$D/server_cmd.sh"
  chmod +x "$D/server_cmd.sh"
  say "k=$K: starting server (conc '$CONC')"
  docker run -d --name "$NAME" $(dgpu_args) \
    -v "$MODEL":/model:ro -v "$DRAFT":/draft:ro -v "$D":/out \
    -e AITER_LOG_LEVEL=WARNING -e NCCL_IB_DISABLE=1 -e RCCL_MSCCL_ENABLE=1 -e NCCL_DEBUG=WARN \
    "$IMG" bash /out/server_cmd.sh >"$D/container_id.txt" 2>&1 || { say "k=$K: docker run failed"; return 1; }
  for i in $(seq 1 3000); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      say "k=$K: container exited during load. Last lines:"; tail -25 "$D/atom_server.log" 2>/dev/null | tee -a "$STATE"; break
    fi
    (( i % 300 == 0 )) && say "  k=$K still loading (${i}s)"
    sleep 1
  done
  if (( ! ready )); then
    say "k=$K: server not ready"; echo failed >"$D/FAILED"; docker rm -f "$NAME" >/dev/null 2>&1; return 1
  fi
  say "k=$K: server READY"
  for C in $CONC; do
    n=$(( C * 10 ))
    timeout 5400 docker exec "$NAME" python -m atom.benchmarks.benchmark_serving \
        --model /model --backend vllm --base-url "http://localhost:${PORT}" \
        --percentile-metrics ttft,tpot,itl,e2el --dataset-name random --ignore-eos \
        --request-rate inf --random-range-ratio 0.8 --trust-remote-code --max-concurrency "$C" \
        --num-prompts "$n" --random-input-len "$ISL" --random-output-len "$OSL" --save-result \
        --result-dir /out --result-filename "c${C}.json" >"$D/c${C}.log" 2>&1
    rc=$?
    say "k=$K c=$C rc=$rc $($PY -c "import json,sys;d=json.load(open('$D/c$C.json'));print('tok/s %.1f TPOT %.2f ms TTFT %.0f ms done %d'%(d['output_throughput'],d['median_tpot_ms'],d['median_ttft_ms'],d['completed']))" 2>/dev/null || echo 'no json')"
  done
  docker stop -t 30 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1
  sleep 30
}

IFS=';' read -ra A <<<"$ARMS"
for a in "${A[@]}"; do run_arm "${a%%:*}" "${a#*:}"; done
say "sweep done"
exec 9>&-

mkdir -p "$BENCH_ROOT/logs/auto"
flock "$BENCH_ROOT/logs/auto/report.lock" $PY analyze_atom_mtp.py "$OUT" 2>&1 | tee -a "$STATE"
say "analysis done; sync and push"
/orcd/data/orcd/022/benchmarks/sync-and-push.sh "Kimi-K3 InferenceX cross-check: add MI355X ATOM + MTP

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >>"$OUT/push.log" 2>&1
say "push rc=$? (log $OUT/push.log)"
say "ATOM + MTP DONE"
