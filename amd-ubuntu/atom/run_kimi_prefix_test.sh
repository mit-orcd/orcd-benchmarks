#!/usr/bin/env bash
# Prefix-cache test for vLLM recipe 2 (amd-kimi-k3-recipe.pdf) on AMD's 128K/1K workload.
# Question: our 1-4 user numbers are 0.56-0.82x of the PDF's table while 8+ users match. Did AMD's
# low-user runs reuse prompts from the prefix cache? The PDF's client sets no --seed (always seed 0,
# so the same prompts every run) and vLLM v0.29.0 has prefix caching on by default; 10-20 prompts of
# ~122K tokens fit in the 2.8M-token KV cache, so a repeated run can skip most of the prefill.
#
# Same image, server flags and client flags as atom/run_kimi_amdrecipe.sh isl128k. One server:
#   warm-up    4 prompts, ISL 1024, seed 99 (does not put 128K prompts in the cache)
#   c1_cold    C=1, 10 prompts, seed 0  -- first time these prompts are seen
#   c1_repeat  C=1, 10 prompts, seed 0  -- the same prompts again (cache hits if they stayed)
#   c1_seed1   C=1, 10 prompts, seed 1  -- new prompts (back to cold)
#   c2_cold / c2_repeat   C=2, 20 prompts, seed 2
#   c4_cold / c4_repeat   C=4, 40 prompts, seed 3 (4.9M tokens > KV cache: few hits expected)
#   ./launch.sh <node> atom/run_kimi_prefix_test.sh 8 16 ...   -- instead: for each user count C,
#   cC_cold / cC_repeat   C users, 10*C prompts, seed 100+C (used to fill 8-128 users)
# Prefix-cache hit rate per test = delta of vllm:prefix_cache_hits / vllm:prefix_cache_queries
# read from the server's /metrics before and after each test.
#
#   ./launch.sh node6100 atom/run_kimi_prefix_test.sh
# Output: logs/<node>/kimi-amd-recipe/atom/kimi_prefix_test_<ts>/,
#         results/ubuntu/kimi-prefix-cache-test.md (+ marker blocks in kimi-summary.md §2 and
#         mi355x-vs-b200.md §5.3, by analyze_prefix_test.py, which merges all prefix-test runs)
set -uo pipefail
ARM=prefix
export RESULTS_SUBDIR=kimi-amd-recipe
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
export CTR_OVERLAY="vllm-v0.29.0-$(hostname -s)"   # persistent overlay: the AITER JIT build outgrows the tmpfs
cd "$BENCH_ROOT/atom"
ISL=128000; OSL=1000; RR=0.2; BT=36000   # timeout per test; 128 users take ~5 h
IMG="${VLLM_IMG:-vllm/vllm-openai-rocm:v0.29.0}"
MODEL="${MKIMI:-$KIMI_MODEL}"
PORT="${PORT:-8030}"
NAME="vllm-kimi-amdrecipe"
ME=$(id -un)

TS=$(date +%Y%m%d_%H%M%S)
OUT=$LOG_ROOT/atom/kimi_prefix_test_$TS; mkdir -p "$OUT" "$RESULTS"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $HOSTNAME_S $*" | tee -a "$STATE"; }
USERS="$*"
say "prefix-cache test (users: ${USERS:-1 2 4 default set}), vLLM recipe 2, ISL/OSL=$ISL/$OSL rr=$RR image=$IMG out=$OUT"

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
  extra+=("${@:5}")
  timeout "$4" docker exec "$NAME" vllm bench serve \
    --host localhost --port "$PORT" --endpoint /v1/completions \
    --model moonshotai/Kimi-K3 --tokenizer /model --trust-remote-code \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio "$RR" --ignore-eos \
    --max-concurrency "$1" --num-prompts "$2" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 25,50,75,90,95,99 \
    "${extra[@]}"
}

metric() {  # sum of a counter over all label sets, from /metrics
  curl -sf "http://localhost:${PORT}/metrics" | awk -v m="$1" '$1 ~ "^"m"(_total)?([{]|$)" {s += $NF} END {printf "%.0f\n", s}'
}
# $1 tag  $2 concurrency  $3 prompts  $4 seed
test_run() {
  local q0 h0 q1 h1 f="prefix_$1.json"
  q0=$(metric vllm:prefix_cache_queries); h0=$(metric vllm:prefix_cache_hits)
  say "$1: C=$2, $3 prompts, seed $4"
  bench "$2" "$3" "$f" "$BT" --seed "$4" >"$OUT/bench_$1.log" 2>&1; local rc=$?
  q1=$(metric vllm:prefix_cache_queries); h1=$(metric vllm:prefix_cache_hits)
  echo "$1|$2|$3|$4|$((q1 - q0))|$((h1 - h0))|$f" >> "$OUT/tests.txt"
  say "$1: rc=$rc queries=$((q1 - q0)) hits=$((h1 - h0)) $(grep -E 'Total token throughput|Median TTFT|Median TPOT' "$OUT/bench_$1.log" | tr -s ' ' | paste -sd';')"
}

# ---- 5. tests -----------------------------------------------------------------------------
: > "$OUT/tests.txt"
say "warm-up (4 prompts, ISL 1024, seed 99)"
ISL0=$ISL; ISL=1024; bench 4 4 "" 1800 --seed 99 >"$OUT/warmup.log" 2>&1; ISL=$ISL0
say "metrics: $(curl -sf http://localhost:${PORT}/metrics | grep -c '^vllm:prefix_cache') prefix-cache lines"
if [[ -z $USERS ]]; then
  test_run c1_cold   1 10 0
  test_run c1_repeat 1 10 0
  test_run c1_seed1  1 10 1
  test_run c2_cold   2 20 2
  test_run c2_repeat 2 20 2
  test_run c4_cold   4 40 3
  test_run c4_repeat 4 40 3
else
  for C in $USERS; do
    test_run "c${C}_cold"   "$C" $((10 * C)) $((100 + C))
    test_run "c${C}_repeat" "$C" $((10 * C)) $((100 + C))
  done
fi
grep -i "prefix cache hit rate" "$OUT/server.log" | tail -3 >> "$STATE"
stop_server
flock -u 9; exec 9>&-

# ---- 6. analysis --------------------------------------------------------------------------
flock "$BENCH_ROOT/logs/auto/report.lock" $PY analyze_prefix_test.py "$OUT" >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> results/ubuntu/kimi-prefix-cache-test.md"
echo "STATUS=$( [[ -s $OUT/tests.txt ]] && echo ok || echo failed)" >> "$STATE"
echo "$(date -Iseconds) prefix test (users: ${USERS:-1 2 4}) on $HOSTNAME_S: done, results/ubuntu/kimi-prefix-cache-test.md + mi355x-vs-b200.md table 5.3 updated ($OUT)" >> "$BENCH_ROOT/results/status.txt"
say "PREFIX TEST DONE"
