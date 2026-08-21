#!/usr/bin/env bash
# Shared body for the Kimi-K3 runs. Sourced by job-kimi-1node.sh (TP8, PP1) and
# job-kimi-base.sh (TP8 x PP2). Not executable on its own.
#
# Call kimi_run after sourcing common/env.sh from inside a Slurm allocation.
# Reads: TP PP and everything else from common/env.sh.

kimi_run() {
  local TAG="${1:-base}"

  TS=$(date +%Y%m%d_%H%M%S)
  RUN=$LOG_ROOT/kimi_${TAG}_$TS; mkdir -p "$RUN"
  STATE=$RUN/STATE.txt
  say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }

  SRV_DIR=$RUN/server
  SWEEP_DIR=$RUN/sweep

  mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  local NNODES=${#NODES[@]} WORLD=$(( TP * PP ))
  say "Kimi-K3 B200 run [$TAG]: job $SLURM_JOB_ID on ${NODES[*]}"
  say "  TP=$TP PP=$PP world=$WORLD  max_num_seqs=$MAX_NUM_SEQS  max_model_len=$MAX_MODEL_LEN"
  say "  ISL/OSL=$ISL/$OSL  conc='$KIMI_CONC'  gpu_mem_util=$GPU_MEM_UTIL"
  say "  out: $RUN"

  # Whatever happens -- error, walltime kill, scancel -- the server and Ray session come
  # down. Otherwise a killed job leaves GPUs held by orphan workers and the next user's
  # allocation fails for reasons that look like a hardware fault.
  finish() {
    local rc=$?
    say "trap: tearing down (rc=$rc)"
    ./stop-vllm-server.sh "$SRV_DIR" >>"$RUN/stop.log" 2>&1 || true
    exit $rc
  }
  trap finish EXIT TERM INT

  # ---- preflight -------------------------------------------------------------------
  # The allocation must match the requested parallelism exactly, not merely be big
  # enough. Slurm handing over 2 nodes for a PP=1 job would silently leave 8 GPUs idle
  # and bill them; handing over 1 node for PP=2 would fail much later, inside Ray.
  local want_nodes=$(( (WORLD + 7) / 8 ))
  if [[ $NNODES -ne $want_nodes ]]; then
    say "ABORT: TP=$TP x PP=$PP = $WORLD GPUs wants $want_nodes node(s), allocation has $NNODES"
    say "       (check the #SBATCH -N and -w lines against ./notes)"
    exit 1
  fi
  # PP is the inter-node axis and TP the intra-node one; TP must fit inside one node.
  if [[ $TP -gt 8 ]]; then
    say "ABORT: TP=$TP exceeds the 8 GPUs in one node -- TP must stay on NVLink."
    exit 1
  fi
  local ngpu_alloc
  ngpu_alloc=$(srun --overlap -N1 -n1 -w "${NODES[0]}" \
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
  if [[ "${ngpu_alloc:-0}" -ne 8 ]]; then
    say "ABORT: ${NODES[0]} shows ${ngpu_alloc:-0} GPU(s), expected 8 (--gpus-per-node=b200:8)"
    exit 1
  fi
  say "parallelism OK: TP=$TP within each node x PP=$PP across $NNODES node(s) = $WORLD GPUs"

  # Never pull from inside an allocation: the image is fetched exactly once by
  # pull-image.sh on the login node. A GPU job that finds it missing must die now,
  # not spend 20 minutes of a 2-node hold downloading 15 GB.
  if ! check_image; then
    say "ABORT: image not ready. Run ./pull-image.sh on the login node first."
    exit 1
  fi
  say "image OK: $VLLM_SIF ($(stat -c %s "$VLLM_SIF") bytes, manifest verified)"
  # The checkpoint is pre-staged and read-only at $MKIMI -- nothing downloads it. The
  # thing that can actually go wrong is the compute node not mounting the export, so
  # check that from INSIDE the allocation rather than trusting the login node's view.
  local seen
  seen=$(srun --overlap -N1 -n1 -w "${NODES[0]}" bash -c "[[ -d '$MKIMI' ]] && echo yes" 2>/dev/null)
  if [[ "$seen" != "yes" ]]; then
    say "ABORT: ${NODES[0]} cannot see $MKIMI"
    say "       The checkpoint lives on fstor025.ib:/compute/orcd/025, which the login"
    say "       node mounts. If the B200 nodes do not, the export has to be mounted"
    say "       there (ask ORCD) or the weights staged somewhere they can reach."
    exit 1
  fi
  if ! check_model; then
    say "ABORT: staged checkpoint failed its integrity check"
    exit 1
  fi
  local shards bytes
  shards=$(ls -1 "$MKIMI"/*.safetensors | wc -l)
  bytes=$(ls -lL "$MKIMI"/*.safetensors | awk '{t+=$5} END{print t+0}')
  say "model OK (pre-staged, read-only): $MKIMI"
  say "  -> $(readlink -f "$MKIMI")"
  say "  $shards shards, $bytes bytes ($(awk -v b="$bytes" 'BEGIN{printf "%.3f TiB", b/2^40}'))"
  KIMI_ONDISK_BYTES=$bytes

  local n busy
  for n in "${NODES[@]}"; do
    busy=$(srun --overlap -N1 -n1 -w "$n" nvidia-smi \
            --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk '$1>1000{c++} END{print c+0}')
    say "$n: $busy GPU(s) with >1 GiB resident"
    [[ "${busy:-0}" -eq 0 ]] || { say "ABORT: $n has busy GPUs despite --exclusive"; exit 1; }
  done

  # Record the per-GPU HBM the run actually saw, and hand it to the analyzer. The
  # "does not fit on one node" arithmetic is only ~23 GB wide on weights alone, so the
  # report must do it against measured capacity rather than a spec-sheet number.
  HBM_MIB=$(srun --overlap -N1 -n1 -w "${NODES[0]}" nvidia-smi \
    --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "${HBM_MIB:-0}" >"$RUN/hbm_mib.txt"
  local node_gb=$(( ${HBM_MIB:-0} * 8 * 1048576 / 1000000000 ))
  say "per-GPU HBM: ${HBM_MIB:-unknown} MiB; node total ~${node_gb} GB vs 1561 GB of weights"

  # ---- Blackwell launch flags ------------------------------------------------------
  # From the vLLM recipe's blackwell baseline + multi_node_tp_pp override. Not free
  # tuning knobs; each is here for a stated reason:
  #   fastsafetensors               1.42 TiB of weights off shared NFS
  #   no-enable-flashinfer-autotune recipe baseline for Blackwell
  #   attention-config              required companion to fp8 KV on this model
  #   no-enable-prefix-caching      CORRECTNESS: KDA recurrent state is per-request and
  #                                 cannot be rebuilt from the paged MLA cache, so prefix
  #                                 reuse would be silently wrong. Also matches MI355X.
  #   max-num-batched-tokens 8192   caps prefill chunks so one long request cannot OOM a
  #                                 pipeline stage
  local KIMI_ARGS=(
    --load-format fastsafetensors
    --no-enable-flashinfer-autotune
    --no-enable-prefix-caching
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"flashinfer"}'
  )

  # ---- server ----------------------------------------------------------------------
  say "STAGE server (expect a long load: 1.42 TiB, timeout ${READY_TIMEOUT}s)"
  local load_t0 rc
  load_t0=$(date +%s)
  ./run-vllm-server.sh "$MKIMI" "$TP" "$PP" "$PORT" "$SRV_DIR" "${KIMI_ARGS[@]}" \
    2>&1 | tee -a "$RUN/server_launch.log"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    say "server failed to start (rc=$rc)"
    kimi_diagnose_failure "$TAG" "$SRV_DIR" "$RUN" "${HBM_MIB:-0}"
    exit 1
  fi
  local load_s=$(( $(date +%s) - load_t0 ))
  say "server up after ${load_s}s"
  echo "$load_s" >"$RUN/load_seconds.txt"

  local HOST
  HOST=$(cat "$SRV_DIR/server.host")

  # ---- sweep -----------------------------------------------------------------------
  say "STAGE sweep (conc: $KIMI_CONC)"
  ./run-vllm-bench.sh "$MKIMI" "$HOST" "$PORT" "$ISL" "$OSL" "$KIMI_CONC" "$SWEEP_DIR" \
    2>&1 | tee -a "$RUN/bench.log"
  local brc=${PIPESTATUS[0]}
  say "sweep rc=$brc"

  # Capture the run's own configuration before tearing the server down. The report must
  # state the config as the SERVER reported it, not as this script intended it.
  srun --overlap -N1 -n1 -w "$HOST" curl -sf "http://localhost:${PORT}/v1/models" \
    >"$RUN/v1_models.json" 2>/dev/null || true

  say "STAGE teardown"
  ./stop-vllm-server.sh "$SRV_DIR" >>"$RUN/stop.log" 2>&1
  trap - EXIT TERM INT

  # ---- analysis --------------------------------------------------------------------
  [[ $brc -ne 0 ]] && say "sweep failed -- analyzing whatever points completed"

  local npoints
  npoints=$(ls "$SWEEP_DIR"/c*.json 2>/dev/null | wc -l)
  if [[ "$npoints" -eq 0 ]]; then
    say "ABORT: no sweep points at all. Not writing a report full of zeros --"
    say "       that is worse than no report. See $SWEEP_DIR and $SRV_DIR/vllm_server.log"
    exit 1
  fi
  say "STAGE analysis ($npoints concurrency points)"

  local HBM_ARG=()
  [[ "${HBM_MIB:-0}" -gt 0 ]] && HBM_ARG=(--hbm-mib "$HBM_MIB")
  local BASENAME="kimi-k3-base-b200"
  [[ "$TAG" != "base" ]] && BASENAME="kimi-k3-base-b200-$TAG"

  apptainer exec $(apt_args) "$VLLM_SIF" $PY_C analyze-kimi-b200.py \
    --sweep "$SWEEP_DIR" \
    --server-log "$SRV_DIR/vllm_server.log" \
    --model-config "$MKIMI/config.json" \
    --run-dir "$RUN" \
    --tp "$TP" --pp "$PP" --isl "$ISL" --osl "$OSL" \
    --max-num-seqs "$MAX_NUM_SEQS" --kv-dtype "$KV_CACHE_DTYPE" \
    --weight-bytes "${KIMI_ONDISK_BYTES:-$KIMI_BYTES}" \
    --basename "$BASENAME" \
    "${HBM_ARG[@]}" \
    -o "$RESULTS" \
    >"$RUN/analyze.log" 2>&1
  local arc=$?
  sed 's/^/    /' "$RUN/analyze.log" | tee -a "$STATE"
  say "analyze rc=$arc -> $RESULTS/$BASENAME.md"
  say "DONE. run dir: $RUN"
}

# Explain a start-up failure instead of just dumping a stack trace. The single-node
# attempt is EXPECTED to fail this way, and "it OOMed" vs "it crashed" is the whole
# result of that experiment -- so the distinction has to be made explicitly.
kimi_diagnose_failure() {
  local TAG=$1 SRV_DIR=$2 RUN=$3 HBM_MIB=$4
  local log="$SRV_DIR/vllm_server.log"
  say "---- failure diagnosis ----"
  if [[ -f "$log" ]] && grep -qiE "out of memory|OutOfMemoryError|CUDA error: out of memory|NCCL.*alloc" "$log"; then
    local node_gb=$(( HBM_MIB * 8 * 1048576 / 1000000000 ))
    say "VERDICT: out of memory during model load."
    if [[ "$TAG" == "1node" ]]; then
      say "  This is the EXPECTED result and it is the measurement:"
      say "    node HBM  ~${node_gb} GB (8 x ${HBM_MIB} MiB)"
      say "    weights    1561 GB"
      say "  The checkpoint does not fit on one B200 node. Proceed to job-kimi-base.sh"
      say "  (TP8 x PP2 across two nodes), which is the vLLM recipe's verified B200 layout."
    else
      say "  Unexpected at TP=$TP PP=$PP. Check gpu_memory_utilization ($GPU_MEM_UTIL) --"
      say "  the flashinfer MXFP4 MoE kernel takes ~1.6 GiB outside vLLM's pool on the"
      say "  first forward, which is why the recipe uses 0.90 rather than 0.95 here."
    fi
  elif [[ -f "$log" ]] && grep -qiE "mlx5dv_reg_dmabuf_mr|unhandled system error" "$log"; then
    say "VERDICT: NCCL dmabuf registration failed."
    say "  Set NCCL_DMABUF_ENABLE=0 (falls back to nvidia_peermem, still GPUDirect RDMA)."
    say "  This is a documented failure mode in the vLLM Kimi-K3 recipe."
  elif [[ -f "$log" ]] && grep -qiE "no kernel image|CUDA driver version is insufficient|forward compatibility" "$log"; then
    say "VERDICT: driver/CUDA mismatch."
    say "  The vllm:kimi-k3 image is a cu130 build and needs an r580+ host driver."
    say "  Run ./job-probe-drivers.sh and, if this node is on r570, move to the Rocky"
    say "  nodes (\$B200_NODES_ALT) which were on 590.48.01."
  else
    say "VERDICT: unrecognized failure -- see the tail below and $log"
  fi
  say "---- last 40 lines of the server log ----"
  tail -40 "$log" 2>/dev/null | sed 's/^/    /' | tee -a "$STATE"
}
