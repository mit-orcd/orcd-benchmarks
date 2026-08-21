#!/usr/bin/env bash
# IMPROVEMENT EXPERIMENTS: test the four levers from section 3.2 of
# results/kimi-k3-base-b200.md ("How to improve it").
#
# Submit with:  sbatch job-improve-b200.sh
#
# Writes ONLY new files:
#   logs/improve_<ts>/...            per-arm server logs + sweeps
#   results/kimi-k3-improve-b200.md  the summary
# It never touches results/kimi-k3-base-b200.{md,csv} or RUN-SUMMARY.md, and it does
# not modify any existing script -- it drives run-vllm-server.sh / run-vllm-bench.sh /
# stop-vllm-server.sh purely through their documented env vars and arguments.
#
# All arms run inside ONE allocation, sequentially, so the 2-node hold is paid once
# rather than four times. Each arm restarts the server because each changes engine
# config (~11 min load per arm).
#
#SBATCH -p mit_testing
#SBATCH --reservation=rres_joohye_2026-08-20_lj4j2ya3
#SBATCH -A rres_acc_joohye_2026-08-20_lj4j2ya3
#SBATCH -w node5700-c1,node5701-c1
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=b200:8
#SBATCH --mem=0
#SBATCH -t 06:00:00
#SBATCH -J kimi-improve
#SBATCH --exclusive
#SBATCH -o out/kimi-improve.%J.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source common/env.sh
module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null

TS=$(date +%Y%m%d_%H%M%S)
RUN=$LOG_ROOT/improve_$TS; mkdir -p "$RUN"
STATE=$RUN/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
say "IMPROVE run: job $SLURM_JOB_ID on ${NODES[*]}"
say "  baseline for comparison: max_num_seqs=64, peak 1696.4 tok/s @ c=64 (job 20916742)"

CUR_SRV=""
finish() {
  local rc=$?
  say "trap: tearing down (rc=$rc)"
  [[ -n "$CUR_SRV" ]] && ./stop-vllm-server.sh "$CUR_SRV" >>"$RUN/stop.log" 2>&1 || true
  exit $rc
}
trap finish EXIT TERM INT

# Blackwell flags, identical to the baseline run so the ONLY difference per arm is the
# lever under test. Copied here rather than imported so this script cannot perturb the
# baseline's definition.
BASE_ARGS=(
  --load-format fastsafetensors
  --no-enable-flashinfer-autotune
  --no-enable-prefix-caching
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"flashinfer"}'
)

# run_arm <tag> <max_num_seqs> <conc_list> [extra server args...]
run_arm() {
  local tag=$1 mns=$2 conc=$3; shift 3
  local extra=("$@")
  local sdir="$RUN/${tag}_server" wdir="$RUN/${tag}_sweep"
  say "===== ARM $tag : max_num_seqs=$mns conc='$conc' extra='${extra[*]}' ====="

  export MAX_NUM_SEQS="$mns"
  local t0; t0=$(date +%s)
  CUR_SRV="$sdir"
  ./run-vllm-server.sh "$MKIMI" "$TP" "$PP" "$PORT" "$sdir" \
    "${BASE_ARGS[@]}" "${extra[@]}" >"$RUN/${tag}_launch.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    say "ARM $tag: server FAILED to start (rc=$rc)"
    # Capture the reason verbatim -- for the spec-decode arm the failure IS the result.
    grep -oE "(ValueError|NotImplementedError|RuntimeError|AssertionError|TypeError): .{0,220}" \
      "$sdir/vllm_server.log" 2>/dev/null | sort -u | head -5 \
      | sed 's/^/    /' | tee -a "$STATE"
    echo "FAILED_TO_START rc=$rc" >"$RUN/${tag}.status"
    ./stop-vllm-server.sh "$sdir" >>"$RUN/stop.log" 2>&1; CUR_SRV=""
    return 1
  fi
  say "ARM $tag: server up after $(( $(date +%s) - t0 ))s"

  local host; host=$(cat "$sdir/server.host")
  ./run-vllm-bench.sh "$MKIMI" "$host" "$PORT" "$ISL" "$OSL" "$conc" "$wdir" \
    >"$RUN/${tag}_bench.log" 2>&1
  local brc=$?
  say "ARM $tag: sweep rc=$brc, $(ls "$wdir"/c*.json 2>/dev/null | wc -l) point(s)"
  echo "OK sweep_rc=$brc" >"$RUN/${tag}.status"
  ./stop-vllm-server.sh "$sdir" >>"$RUN/stop.log" 2>&1; CUR_SRV=""
  sleep 20   # let VRAM drain before the next arm
  return 0
}

# ---- LEVER 1: raise --max-num-seqs -------------------------------------------------
# The headline recommendation. At the baseline cap of 64 each expert sees only 1.7
# tokens; raising the cap widens those GEMMs, which is the direct attack on the
# latency-bound (not bandwidth-bound) decode path.
run_arm lever1_mns256 256 "64 128 256"
run_arm lever1_mns512 512 "128 256 512"

# ---- LEVER 3: expert parallelism ---------------------------------------------------
# Run at max_num_seqs=256 and swept over the SAME concurrencies as lever1_mns256, so
# EP is the only variable. The MI355X work learned this the hard way: an EP arm at
# cap 256 was compared against a TP-only arm at cap 64, and only the single c=64 row
# was a valid comparison. Matched caps here from the start.
run_arm lever3_ep 256 "64 128 256" --enable-expert-parallel

# ---- LEVER 2: speculative decoding (DSpark) ----------------------------------------
# The vLLM recipe gates DSpark OFF the multi_node_tp_pp profile (does not compose with
# pipeline parallelism, vllm-project/vllm#50098) -- and PP is not optional on B200,
# since the checkpoint does not fit one node. This arm establishes that by measurement
# rather than by citation. A short timeout: if it is going to be rejected, it is
# rejected at config time, long before any weight load.
say "===== ARM lever2_spec : capability probe (expected to be rejected) ====="
READY_TIMEOUT=900 run_arm lever2_spec 64 "64" \
  --speculative-config '{"model":"RedHatAI/Kimi-K3-speculator.dspark","num_speculative_tokens":8,"method":"dspark","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
  || say "lever2_spec: did not start (see above) -- this is the expected outcome"

# ---- LEVER 4: prefill/decode disaggregation ----------------------------------------
# Not run, and the arithmetic is the reason rather than a preference. P/D disagg needs
# TWO independent engine instances (a prefill pool and a decode pool), each holding the
# full 1561 GB of weights. One instance already needs 16 GPUs / 2 nodes just to fit, so
# a P/D cluster needs >= 32 GPUs / 4 nodes. This allocation has 2.
{
  echo "not_run"
  echo "reason=needs >=2 full engine instances; 1561 GB each; 16 GPUs/instance => >=32 GPUs (4 nodes)"
  echo "allocated_nodes=${#NODES[@]}"
  echo "allocated_gpus=$(( ${#NODES[@]} * 8 ))"
} >"$RUN/lever4_pd.status"
say "LEVER 4 (P/D disagg): NOT RUN -- needs >=4 nodes (2 instances x 16 GPUs); have ${#NODES[@]}"

trap - EXIT TERM INT

# ---- analysis ----------------------------------------------------------------------
say "STAGE analysis"
apptainer exec $(apt_args) "$VLLM_SIF" $PY_C analyze-improve-b200.py \
  --run-dir "$RUN" -o "$RESULTS" \
  --baseline-sweep "$LOG_ROOT/kimi_base_20260821_130024/sweep" \
  >"$RUN/analyze.log" 2>&1
say "analyze rc=$? -> $RESULTS/kimi-k3-improve-b200.md"
sed 's/^/    /' "$RUN/analyze.log" | tail -5 | tee -a "$STATE"
say "DONE. run dir: $RUN"
