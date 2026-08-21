#!/usr/bin/env bash
# Concurrency sweep against an already-running vLLM server.
#
# Usage: ./run-vllm-bench.sh <model_dir> <HOST> <PORT> <ISL> <OSL> <CONC_LIST> <OUTDIR>
#
# Calls `vllm bench serve`, which is the same code lineage as the
# `atom.benchmarks.benchmark_serving` the MI355X run used and emits the same JSON keys --
# which is what lets one analyzer read both runs and compare them directly.
#
# It MUST be the `vllm bench serve` CLI, not `python3 -m vllm.benchmarks.serve`:
# vllm/benchmarks/serve.py has no `if __name__ == "__main__"` guard, so running it as a
# module silently does NOTHING and exits 0 -- no benchmark, no JSON, no error. That would
# have produced an empty sweep only AFTER paying the ~30 minute 1.42 TiB weight load.
# All 19 flags below were checked against `vllm bench serve --help=all` (the plain
# --help prints only group names).
#
# One c<N>.json + one c<N>.log per concurrency point, written as each point finishes,
# so a walltime kill still leaves an analyzable partial sweep.
#
# RAYON_NUM_THREADS is set for the same reason vLLM's own bench CLI caps tokio: the
# binary logs "capping tokio worker threads ... available_parallelism=224
# capped_worker_threads=32", but its SEPARATE rayon-core pool has no such cap and tries
# to spawn all 224 OS threads unconditionally. On a node already running a TP8xPP2
# server (16 GPU worker processes, each with many CUDA/NCCL/inductor threads), that
# spawn hit pthread_create() EAGAIN and the bench binary panicked and aborted (rc=134)
# at EVERY concurrency point, before sending a single request (job 20907074). Capping
# rayon the same way tokio caps itself avoids the race for the node's thread budget.
set -uo pipefail
cd "$(dirname "$0")"
source common/env.sh

MODEL="${1:?usage: $0 <model_dir> <HOST> <PORT> <ISL> <OSL> <CONC> <OUTDIR>}"
HOST="${2:?host}"
PORTn="${3:-8000}"
ISLn="${4:-1024}"
OSLn="${5:-1024}"
CONC_LIST="${6:-1 2 4 8 16 32 64}"
OUT="${7:?outdir}"

mkdir -p "$OUT"
SUM="$OUT/summary.txt"
module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null
check_image || exit 1

srun --overlap -N1 -n1 -w "$HOST" curl -sf "http://localhost:${PORTn}/v1/models" >/dev/null 2>&1 || {
  echo "ERROR: no vLLM server responding on $HOST:$PORTn" >&2; exit 1; }

{ echo "vLLM serving sweep $(date -Iseconds)"
  echo "model    : $MODEL"
  echo "served   : $SERVED_NAME"
  echo "host:port: $HOST:$PORTn"
  echo "TP/PP    : $TP/$PP    max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN"
  echo "ISL/OSL  : $ISLn/$OSLn  range_ratio=$RANDOM_RANGE_RATIO  seed=$SEED"
  echo "conc     : $CONC_LIST"
  echo "image    : $VLLM_SIF"
  echo
  printf '%6s %12s %12s %12s %12s\n' conc req/s out_tok/s ttft_ms_med tpot_ms_med
} | tee "$SUM"

for C in $CONC_LIST; do
  n=$(( C * PROMPT_MULTIPLIER ))
  log="$OUT/c${C}.log"; json="$OUT/c${C}.json"
  start=$(date +%s)
  timeout "$BENCH_TIMEOUT" srun --overlap -N1 -n1 -w "$HOST" \
    apptainer exec $(apt_args) \
    --env RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-16}" \
    "$VLLM_SIF" \
    vllm bench serve \
      --backend openai \
      --base-url "http://localhost:${PORTn}" \
      --model "$MODEL" \
      --served-model-name "$SERVED_NAME" \
      --dataset-name random \
      --random-input-len "$ISLn" \
      --random-output-len "$OSLn" \
      --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --max-concurrency "$C" \
      --num-prompts "$n" \
      --num-warmups "$(( C * 2 ))" \
      --request-rate inf \
      --ignore-eos \
      --trust-remote-code \
      --seed "$SEED" \
      --save-result --result-filename "$json" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,99 \
    >"$log" 2>&1
  rc=$?; dur=$(( $(date +%s) - start ))

  if [[ -f "$json" ]]; then
    read -r rps ots ttft tpot completed < <(python3 - "$json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("request_throughput", 0), d.get("output_throughput", 0),
      d.get("median_ttft_ms", 0), d.get("median_tpot_ms", 0), d.get("completed", 0))
PY
)
    printf '%6s %12.2f %12.1f %12.1f %12.2f   rc=%s %ss completed=%s\n' \
      "$C" "$rps" "$ots" "$ttft" "$tpot" "$rc" "$dur" "$completed" | tee -a "$SUM"

    # `vllm bench serve` exits 0 even when EVERY request fails -- it warns and writes
    # zeros. Without this check a dead server yields a full sweep of 0.00 rows and a
    # cheerful rc=0. Same trap ATOM's script guards, same response: stop immediately
    # rather than burn the remaining points.
    if [[ "${completed:-0}" -eq 0 ]]; then
      { echo "FATAL: 0 requests completed at concurrency $C."
        # Say WHICH failure it was rather than guessing. The first version asserted a
        # served-model-name mismatch; the actual cause at c=16 was the engine dying
        # (HTTP 500 EngineDeadError, then connection refused), and the wrong hint sent
        # the investigation to the wrong place.
        if grep -q "EngineDeadError\|EngineCore encountered an issue" "$log" 2>/dev/null; then
          echo "       CAUSE: the ENGINE died (HTTP 500 / EngineDeadError)."
          echo "       The server crashed mid-sweep -- the root cause is in the SERVER log,"
          echo "       not this client log. Look for the first traceback in:"
          echo "         $(dirname "$OUT")/server/vllm_server.log"
        elif grep -q "Connection refused" "$log" 2>/dev/null; then
          echo "       CAUSE: connection refused -- the server process is gone."
        elif grep -qE "HTTP 4[0-9][0-9]" "$log" 2>/dev/null; then
          echo "       CAUSE: HTTP 4xx. A served-model-name mismatch returns 400 per"
          echo "       request while /v1/models still answers 200, so readiness passes"
          echo "       and every metric records 0."
        else
          echo "       CAUSE: unrecognized -- inspect $log."
        fi
        echo "       Aborting sweep."; } | tee -a "$SUM"
      exit 1
    fi
  else
    printf '%6s %12s %12s %12s %12s   rc=%s %ss (no json)\n' \
      "$C" - - - - "$rc" "$dur" | tee -a "$SUM"
    [[ $rc -eq 124 ]] && echo "       (timed out after ${BENCH_TIMEOUT}s)" | tee -a "$SUM"
  fi
done

echo "results: $OUT" | tee -a "$SUM"
echo "$OUT" >"$LOG_ROOT/CURRENT_SWEEP_DIR.txt"
