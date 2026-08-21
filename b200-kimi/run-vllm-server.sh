#!/usr/bin/env bash
# Bring up the vLLM OpenAI server for a model, inside the allocation.
#
# Usage: ./run-vllm-server.sh <model_dir> <TP> <PP> <PORT> <OUTDIR> [extra vllm args...]
#
# Must run from INSIDE an sbatch allocation (it uses $SLURM_JOB_NODELIST). For PP>1 it
# stands up a Ray cluster across the allocated nodes first, because vLLM's default
# multiprocessing executor is single-node only.
#
# Keeps the ATOM scripts' refuse-don't-kill discipline: it never pkills anything, only
# ever tears down the Ray session and container instances this job created.
#
# Flag notes -- checked against vLLM main, not guessed:
#   * `--tensor-parallel-size` / `--pipeline-parallel-size` (there is no `-pp` alias).
#   * `--kv-cache-dtype` is spelled with DASHES here (ATOM used underscores).
#   * The port flag is `--port` (ATOM used `--server-port`).
#   * `--served-model-name` IS set here, and the bench client is told the same string.
#     ATOM's script deliberately omitted it because its client passed the model *path*;
#     `vllm bench serve` takes `--served-model-name` explicitly, so we can name it and
#     keep client and server aligned that way instead.
set -uo pipefail
cd "$(dirname "$0")"
source common/env.sh

MODEL="${1:?usage: $0 <model_dir> <TP> <PP> <PORT> <OUTDIR> [extra args...]}"
TPn="${2:-8}"
PPn="${3:-1}"
PORTn="${4:-8000}"
OUT="${5:?outdir}"
shift 5 2>/dev/null || true
EXTRA=("$@")

mkdir -p "$OUT"
SRVLOG="$OUT/vllm_server.log"
say() { echo "[$(date -Iseconds)] server: $*"; }

[[ -d "$MODEL" ]] || { echo "ERROR: model dir not found: $MODEL" >&2; exit 1; }
check_image || exit 1
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: not inside a Slurm allocation" >&2; exit 1; }

module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${NODES[0]}"
NNODES=${#NODES[@]}
WORLD=$(( TPn * PPn ))
say "nodes=${NODES[*]} head=$HEAD world=$WORLD (TP=$TPn PP=$PPn)"

if [[ $(( NNODES * 8 )) -lt $WORLD ]]; then
  echo "ERROR: TPxPP=$WORLD needs $(( (WORLD + 7) / 8 )) nodes, allocation has $NNODES" >&2
  exit 1
fi

# Ray's temp dir goes on node-local /dev/shm, NOT on the shared tree. Two reasons:
# a Unix socket path is capped at 107 bytes and ray appends ~40 chars of session and
# socket names to whatever it is given, and putting the object store on NFS would put
# every Ray control message through the filesystem.
RAY_TMP="/dev/shm/ray-${SLURM_JOB_ID}"
RAY_PORT="${RAY_PORT:-6379}"
# Must be a routable IPv4 -- see resolve_ipv4() in common/env.sh for why plain
# `getent hosts` is not safe here (it handed Ray an IPv6 link-local on the compute node
# and the 2-node cluster never came up).
if ! HEAD_IP=$(resolve_ipv4 "$HEAD"); then
  echo "ERROR: could not resolve a routable IPv4 for head node $HEAD" >&2
  getent ahostsv4 "$HEAD" >&2 || true
  exit 1
fi
say "ray head at $HEAD_IP:$RAY_PORT temp=$RAY_TMP"

ENVFILE="$OUT/server.env"
: >"$ENVFILE"
nccl_env >>"$ENVFILE"
cat >>"$ENVFILE" <<ENVEOF
VLLM_ENGINE_READY_TIMEOUT_S=$READY_TIMEOUT
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
VLLM_ALLREDUCE_USE_FLASHINFER=1
HF_HUB_OFFLINE=1
HF_HOME=$HF_HOME
PYTHONUNBUFFERED=1
ENVEOF
APPENV=()
while IFS='=' read -r k v; do [[ -n "$k" ]] && APPENV+=(--env "$k=$v"); done <"$ENVFILE"

CLEANED=0
cleanup() {
  [[ $CLEANED -eq 1 ]] && return; CLEANED=1
  say "cleanup: stopping ray on ${NODES[*]}"
  for n in "${NODES[@]}"; do
    srun --overlap -N1 -n1 -w "$n" \
      apptainer exec $(apt_args) "$VLLM_SIF" $RAY_CLI stop --force >/dev/null 2>&1 || true
    srun --overlap -N1 -n1 -w "$n" rm -rf "$RAY_TMP" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

# --- Ray cluster (only when PP or TP spans more than one node) --------------------
if [[ $WORLD -gt 8 ]]; then
  say "starting ray head on $HEAD"
  srun --overlap -N1 -n1 -w "$HEAD" \
    apptainer exec $(apt_args) "${APPENV[@]}" "$VLLM_SIF" \
    $RAY_CLI start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
              --temp-dir="$RAY_TMP" --num-gpus=8 --block \
    >"$OUT/ray_head.log" 2>&1 &
  RAY_HEAD_PID=$!

  # Wait for the head's GCS to answer before any worker tries to attach; a worker that
  # races the head dies immediately and Slurm reports a confusing 16-vs-8 GPU count.
  for i in $(seq 1 120); do
    srun --overlap -N1 -n1 -w "$HEAD" bash -c "</dev/tcp/$HEAD_IP/$RAY_PORT" 2>/dev/null && break
    sleep 1
  done

  for n in "${NODES[@]:1}"; do
    say "starting ray worker on $n"
    srun --overlap -N1 -n1 -w "$n" \
      apptainer exec $(apt_args) "${APPENV[@]}" "$VLLM_SIF" \
      $RAY_CLI start --address="$HEAD_IP:$RAY_PORT" --temp-dir="$RAY_TMP" --num-gpus=8 --block \
      >"$OUT/ray_worker_$n.log" 2>&1 &
  done

  # Poll for the real GPU count rather than sleeping a fixed interval: a short sleep
  # races weight-free worker registration and vllm then plans for the wrong world size.
  say "waiting for ray to report $WORLD GPUs"
  ok=0
  # Same reasoning as the readiness poll below: run this on the batch host directly when
  # that host is the ray head, so 300 polls do not become 300 Slurm job steps.
  ray_gpus() {
    if [[ "$(hostname -s)" == "${HEAD%%.*}" ]]; then
      apptainer exec $(apt_args) "${APPENV[@]}" --env RAY_ADDRESS="$HEAD_IP:$RAY_PORT" \
        "$VLLM_SIF" $PY_C -c "
import ray;ray.init(address='auto',logging_level='ERROR');print(int(ray.cluster_resources().get('GPU',0)))
" 2>/dev/null | tail -1
    else
      srun --overlap -N1 -n1 -w "$HEAD" \
        apptainer exec $(apt_args) "${APPENV[@]}" --env RAY_ADDRESS="$HEAD_IP:$RAY_PORT" \
        "$VLLM_SIF" $PY_C -c "
import ray;ray.init(address='auto',logging_level='ERROR');print(int(ray.cluster_resources().get('GPU',0)))
" 2>/dev/null | tail -1
    fi
  }
  for i in $(seq 1 300); do
    ngpu=$(ray_gpus)
    if [[ "${ngpu:-0}" -ge "$WORLD" ]]; then ok=1; say "ray has $ngpu GPUs"; break; fi
    (( i % 15 == 0 )) && say "  ray GPUs=${ngpu:-0} after ${i}s"
    sleep 2
  done
  [[ $ok -eq 1 ]] || { echo "ERROR: ray never reached $WORLD GPUs" >&2; tail -20 "$OUT"/ray_*.log >&2; exit 1; }
  EXEC_BACKEND=(--distributed-executor-backend ray)
  APPENV+=(--env RAY_ADDRESS="$HEAD_IP:$RAY_PORT")
else
  EXEC_BACKEND=()
fi

# --- launch the server -------------------------------------------------------------
# Written to a FILE and mounted rather than inlined into `bash -c`, for the same reason
# ATOM's script did: --attention-config takes a JSON argument full of double quotes, and
# nesting that through srun + apptainer + bash -c quoting is how you get a silently
# mangled flag rather than a loud error.
CMD="$OUT/server_cmd.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -uo pipefail'
  echo "exec $PY_C -m vllm.entrypoints.openai.api_server \\"
  echo "  --model $MODEL \\"
  echo "  --served-model-name $SERVED_NAME \\"
  echo "  --tensor-parallel-size $TPn \\"
  echo "  --pipeline-parallel-size $PPn \\"
  echo "  --port $PORTn \\"
  echo "  --gpu-memory-utilization $GPU_MEM_UTIL \\"
  echo "  --max-model-len $MAX_MODEL_LEN \\"
  echo "  --max-num-seqs $MAX_NUM_SEQS \\"
  echo "  --kv-cache-dtype $KV_CACHE_DTYPE \\"
  echo '  --trust-remote-code \'
  for a in "${EXEC_BACKEND[@]}" "${EXTRA[@]}"; do printf '  %q \\\n' "$a"; done
  echo "  2>&1 | tee $SRVLOG"
} >"$CMD"
chmod +x "$CMD"
say "cmd: $CMD"
sed 's/^/    | /' "$CMD"

srun --overlap -N1 -n1 -w "$HEAD" \
  apptainer exec $(apt_args) "${APPENV[@]}" "$VLLM_SIF" bash "$CMD" \
  >"$OUT/server_stdout.log" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" >"$OUT/server.pid"
echo "$HEAD"    >"$OUT/server.host"
echo "$PORTn"   >"$OUT/server.port"

# --- readiness: HTTP *and* real VRAM ------------------------------------------------
# Both checks, like ATOM's. /v1/models answering 200 is not proof the weights are
# resident -- and with 1.42 TiB coming off NFS the gap between "process is up" and
# "model is loaded" is tens of minutes, not seconds.
say "waiting for readiness (timeout ${READY_TIMEOUT}s)"
# Poll LOCALLY, not through srun. sbatch runs the batch script on the first node of the
# allocation, which is exactly $HEAD, so curl/nvidia-smi reach the server directly.
#
# The first version wrapped every single poll in `srun --overlap`, which creates a Slurm
# JOB STEP per iteration -- ~2400 steps for one 40-minute model load, all of them landing
# on the controller and in sacct. On a shared scheduler that is rude and pointless.
# Confirmed in job 20852177: dozens of `curl FAILED exit 7` steps within the first minute.
poll_here=1
[[ "$(hostname -s)" == "${HEAD%%.*}" ]] || poll_here=0
[[ $poll_here -eq 1 ]] || say "  (batch host is not $HEAD -- falling back to srun polling)"

probe_ready() {
  if [[ $poll_here -eq 1 ]]; then
    curl -sf "http://localhost:${PORTn}/v1/models" >/dev/null 2>&1
  else
    srun --overlap -N1 -n1 -w "$HEAD" curl -sf "http://localhost:${PORTn}/v1/models" >/dev/null 2>&1
  fi
}
probe_vram() {
  if [[ $poll_here -eq 1 ]]; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk '$1>10000{n++} END{print n+0}'
  else
    srun --overlap -N1 -n1 -w "$HEAD" \
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk '$1>10000{n++} END{print n+0}'
  fi
}

ready=0
for i in $(seq 1 "$READY_TIMEOUT"); do
  if probe_ready; then
    used=$(probe_vram)
    if [[ "${used:-0}" -ge "$TPn" ]]; then
      say "READY on $HEAD:$PORTn (VRAM resident on $used GPU(s))"
      ready=1; break
    fi
  fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "ERROR: server exited before becoming ready. Last 60 lines:" >&2
    tail -60 "$SRVLOG" "$OUT/server_stdout.log" 2>/dev/null >&2
    exit 1
  fi
  (( i % 60 == 0 )) && say "  still loading, ${i}s elapsed"
  sleep 1
done
[[ $ready -eq 1 ]] || {
  echo "ERROR: not ready within ${READY_TIMEOUT}s. Last 60 lines:" >&2
  tail -60 "$SRVLOG" "$OUT/server_stdout.log" 2>/dev/null >&2
  exit 1
}

# The trap must NOT fire on success -- the caller keeps the server up for the sweep.
trap - EXIT
echo "$OUT" >"$LOG_ROOT/CURRENT_SERVER_DIR.txt"
exit 0
