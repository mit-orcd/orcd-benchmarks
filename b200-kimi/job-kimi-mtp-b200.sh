#!/usr/bin/env bash
# MTP (DSpark speculative decoding) on Kimi-K3, 2 x 8 B200, TP8 x PP2.
# Same server and workload as job-kimi-base.sh (ISL/OSL 1024/1024, max-num-seqs 64,
# conc 1..64, 10 x conc prompts) plus --speculative-config, so the only change is MTP.
#
# Submit:  sbatch job-kimi-mtp-b200.sh      (post job: job-kimi-mtp-post.sh)
#
# Draft model: Inferact/Kimi-K3-DSpark, the same file the MI355X vLLM recipe 1 run uses
# (../amd-software/models/Kimi-K3-DSpark, 7.1 GB). models/Kimi-K3-DSpark-pard/ is a
# symlinked copy whose config.json adds pard_token = mask_token_id (the shim from the
# SemiAnalysis InferenceX B200 recipe; vLLM's parallel drafter reads pard_token).
#
# Not the InferenceX flags --decode-context-parallel-size 8 / --dcp-comm-backend a2a:
# this image (vllm/vllm-openai:kimi-k3) refuses them with a K3 DSpark draft
# ("MLA DSpark does not currently support decode context parallelism",
# vllm/config/speculative.py), so DCP stays 1.
#
# Arms, one server each, run in order inside one allocation:
#   mtp7   draft length 7, attention_backend TOKENSPEED_MLA   (InferenceX B200 setting)
#   mtp7d  draft length 7, default draft attention backend    (only if mtp7 does not start)
#   mtp3   draft length 3, same backend as the arm that started
#SBATCH -p mit_testing
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=b200:8
#SBATCH --exclude=node5500-c1,node5802-c1
#SBATCH --mem=0
#SBATCH -t 06:00:00
#SBATCH -J kimi-mtp-b200
#SBATCH --exclusive
#SBATCH -o out/kimi-mtp-b200.%J.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source common/env.sh
module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null

# The draft's weights are symlinks into amd-software/models, which apt_args does not bind.
export APPTAINER_BIND="/orcd/data/orcd/022/benchmarks/amd-software/models:/orcd/data/orcd/022/benchmarks/amd-software/models:ro"
DRAFT=$BENCH_ROOT/models/Kimi-K3-DSpark-pard
export TP=8 PP=2

TS=$(date +%Y%m%d_%H%M%S)
RUN=$LOG_ROOT/mtp_$TS; mkdir -p "$RUN"
echo "$RUN" >"$OUT_DIR/mtp-run-$SLURM_JOB_ID.txt"
STATE=$RUN/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
say "MTP run: job $SLURM_JOB_ID on ${NODES[*]}, draft $DRAFT"
say "  ISL/OSL=$ISL/$OSL conc='$KIMI_CONC' max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN"

[[ -s $DRAFT/model.safetensors && -s $DRAFT/config.json ]] || { say "ABORT: draft missing: $DRAFT"; exit 1; }
check_image || { say "ABORT: image not ready"; exit 1; }
for n in "${NODES[@]}"; do
  busy=$(srun --overlap -N1 -n1 -w "$n" nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
         | awk '$1>1000{c++} END{print c+0}')
  [[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $n has busy GPUs"; exit 1; }
done
srun --overlap -N1 -n1 -w "${NODES[0]}" nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader | head -1 | sed 's/^/  driver, HBM: /' | tee -a "$STATE"

CUR_SRV=""
finish() {
  local rc=$?
  say "trap: tearing down (rc=$rc)"
  [[ -n "$CUR_SRV" ]] && ./stop-vllm-server.sh "$CUR_SRV" >>"$RUN/stop.log" 2>&1 || true
  exit $rc
}
trap finish EXIT TERM INT

# Identical to lib/kimi-run.sh (the no-spec baseline).
BASE_ARGS=(
  --load-format fastsafetensors
  --no-enable-flashinfer-autotune
  --no-enable-prefix-caching
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"flashinfer"}'
)

# $1 tag  $2 draft length  $3 attention backend for the draft ("" = default)
# returns 2 if the server did not start, 0 if the sweep ran
arm() {
  local tag=$1 k=$2 ab=$3 spec srv sweep host rc
  spec="{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$k,\"method\":\"dspark\""
  [[ -n $ab ]] && spec+=",\"attention_backend\":\"$ab\""
  spec+=",\"draft_sample_method\":\"probabilistic\",\"draft_load_config\":{\"load_format\":\"safetensors\"}}"
  srv=$RUN/$tag/server; sweep=$RUN/$tag/sweep; mkdir -p "$srv" "$sweep"
  echo "$spec" >"$RUN/$tag/spec.json"
  say "ARM $tag: draft length $k, attention_backend ${ab:-default}"
  CUR_SRV=$srv
  ./run-vllm-server.sh "$MKIMI" "$TP" "$PP" "$PORT" "$srv" "${BASE_ARGS[@]}" --speculative-config "$spec" \
    >"$RUN/$tag/server_launch.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    say "ARM $tag: server did not start (rc=$rc). Last lines:"
    grep -h -E 'Error|error|raise|Exception' "$srv/vllm_server.log" "$srv/server_stdout.log" 2>/dev/null | tail -6 | sed 's/^/    /' | tee -a "$STATE"
    ./stop-vllm-server.sh "$srv" >>"$RUN/stop.log" 2>&1; CUR_SRV=""
    echo "server failed" >"$RUN/$tag/FAILED"
    return 2
  fi
  host=$(cat "$srv/server.host")
  say "ARM $tag: server up, sweep $KIMI_CONC"
  ./run-vllm-bench.sh "$MKIMI" "$host" "$PORT" "$ISL" "$OSL" "$KIMI_CONC" "$sweep" >"$RUN/$tag/bench.log" 2>&1
  say "ARM $tag: sweep rc=$?"
  sed 's/^/    /' "$sweep/summary.txt" | tail -8 | tee -a "$STATE"
  grep -h 'Mean acceptance length' "$srv/vllm_server.log" 2>/dev/null | tail -2 | sed 's/^/    /' | tee -a "$STATE"
  ./stop-vllm-server.sh "$srv" >>"$RUN/stop.log" 2>&1; CUR_SRV=""
  return 0
}

AB=TOKENSPEED_MLA
if ! arm mtp7 7 "$AB"; then
  AB=""
  arm mtp7d 7 "" || { say "no MTP server started; stopping"; trap - EXIT TERM INT; say "MTP run DONE (failed)"; exit 1; }
fi
arm mtp3 3 "$AB"

trap - EXIT TERM INT
say "MTP run DONE. run dir: $RUN"
