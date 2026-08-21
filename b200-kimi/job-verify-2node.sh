#!/usr/bin/env bash
# Verify the two things that broke the first attempts, on real hardware, WITHOUT paying
# the 1.42 TiB weight load:
#
#   1. Ray forms a 2-node / 16-GPU cluster.   (job 20852178 died here: `getent hosts`
#      handed ray an IPv6 link-local address and the GCS was unreachable.)
#   2. vLLM can INSPECT KimiK3ForConditionalGeneration, which triggers the Triton JIT
#      compile of cuda_utils.c.               (job 20852177 died here: the host's Spack
#      gcc leaked in via PATH and failed against the container's headers.)
#
# Inspection is the exact step that failed, and it happens before any weight is read, so
# this costs ~10 minutes instead of ~40.
#SBATCH -p mit_testing
#SBATCH --reservation=rres_joohye_2026-08-20_lj4j2ya3
#SBATCH -A rres_acc_joohye_2026-08-20_lj4j2ya3
#SBATCH -w node5700-c1,node5701-c1
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=b200:8
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH -J kimi-verify
#SBATCH --exclusive
#SBATCH -o out/kimi-verify.%J.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source common/env.sh
module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null

TS=$(date +%Y%m%d_%H%M%S)
V=$LOG_ROOT/verify_$TS; mkdir -p "$V"
say() { echo "[$(date -Iseconds)] $*" | tee -a "$V/STATE.txt"; }
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${NODES[0]}"; WORKER="${NODES[1]}"
FAIL=0

say "verify on ${NODES[*]} (job $SLURM_JOB_ID)"

# --- 1. IPv4 resolution, evaluated ON the compute node ------------------------------
say "--- check 1: routable IPv4 for the head node ---"
HEAD_IP=$(resolve_ipv4 "$HEAD") || { say "FAIL: no routable IPv4 for $HEAD"; FAIL=1; }
say "resolve_ipv4($HEAD) = ${HEAD_IP:-NONE}"
say "  (raw getent hosts said: $(getent hosts "$HEAD" | awk '{print $1; exit}') -- the bug)"
case "$HEAD_IP" in
  fe80:*|127.*|169.254.*|"") say "FAIL: unusable head IP"; FAIL=1 ;;
  *) say "OK: head IP is routable IPv4" ;;
esac

# --- 2. Triton JIT / model inspection ------------------------------------------------
say "--- check 2: vLLM can inspect KimiK3 (triggers the Triton compile) ---"
apptainer exec $(apt_args) "$VLLM_SIF" bash -lc '
echo "PATH=$PATH"
echo "gcc -> $(command -v gcc)"
echo "CC=$CC"
'"$PY_C"' -c "
from vllm.engine.arg_utils import EngineArgs
ea = EngineArgs(model=\"'"$MKIMI"'\", trust_remote_code=True,
                tensor_parallel_size=1, pipeline_parallel_size=1,
                max_model_len=1024, load_format=\"dummy\")
cfg = ea.create_engine_config()
print(\"INSPECT_OK\", cfg.model_config.architectures)
"' >"$V/inspect.log" 2>&1
irc=$?
grep -E "^PATH=|^gcc ->|^CC=|INSPECT_OK|libc-header-start|Error in inspecting" "$V/inspect.log" | head -10 | sed 's/^/    /' | tee -a "$V/STATE.txt"
if grep -q "INSPECT_OK" "$V/inspect.log"; then
  say "OK: model inspection succeeded (Triton compile worked)"
else
  say "FAIL: model inspection failed (rc=$irc) -- see $V/inspect.log"
  tail -25 "$V/inspect.log" | sed 's/^/    /' | tee -a "$V/STATE.txt"
  FAIL=1
fi

# --- 2b. FlashInfer JIT under the redirected HOME -------------------------------------
# Job 20856604 loaded all 1.42 TiB, then died in vLLM's memory-profiling pass: FlashInfer
# JIT-builds the TRTLLM FP4 MoE kernel there and guards its cache with a filelock, and
# fcntl.flock() returned EIO on 14/16 ranks because that cache was on NFS. HOME and the
# FlashInfer cache dirs now point at node-local /tmp. Prove the lock works there BEFORE
# spending another 28 minutes of weight load to find out.
say "--- check 2b: filelock works on the redirected JIT cache dirs ---"
apptainer exec $(apt_args) "$VLLM_SIF" $PY_C -c "
import os, fcntl, pathlib
print('HOME =', os.environ.get('HOME'))
print('FLASHINFER_CACHE_DIR =', os.environ.get('FLASHINFER_CACHE_DIR'))
ok = True
for d in [os.environ.get('HOME'), os.environ.get('FLASHINFER_CACHE_DIR'),
          os.environ.get('TRITON_CACHE_DIR')]:
    if not d: continue
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    f = pathlib.Path(d) / '.locktest'
    try:
        fd = os.open(str(f), os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd); f.unlink(missing_ok=True)
        print('FLOCK_OK', d)
    except OSError as e:
        print('FLOCK_FAIL', d, e); ok = False
import flashinfer
print('flashinfer', flashinfer.__version__)
try:
    from flashinfer.jit import core as _c
    real = getattr(_c, 'FLASHINFER_JIT_DIR', None) or getattr(_c, 'FLASHINFER_CACHE_DIR', None)
    print('flashinfer_resolved_cache', real)
    if real and str(real).startswith('/orcd'):
        print('FLOCK_FAIL flashinfer cache still on NFS:', real); ok = False
except Exception as e:
    print('flashinfer cache path probe failed:', e)
print('LOCKTEST_OK' if ok else 'LOCKTEST_FAIL')
" >"$V/locktest.log" 2>&1
grep -E "HOME =|FLASHINFER_CACHE_DIR =|FLOCK_OK|FLOCK_FAIL|flashinfer |LOCKTEST_" "$V/locktest.log" | sed 's/^/    /' | tee -a "$V/STATE.txt"
if grep -q "LOCKTEST_OK" "$V/locktest.log"; then
  say "OK: flock works on every JIT cache dir"
else
  say "FAIL: flock still failing -- see $V/locktest.log"
  FAIL=1
fi

# --- 3. Ray 2-node cluster -----------------------------------------------------------
say "--- check 3: ray forms a 16-GPU cluster across both nodes ---"
RAY_TMP="/dev/shm/ray-verify-$SLURM_JOB_ID"
cleanup() {
  for n in "${NODES[@]}"; do
    srun --overlap -N1 -n1 -w "$n" apptainer exec $(apt_args) "$VLLM_SIF" \
      $RAY_CLI stop --force >/dev/null 2>&1 || true
    srun --overlap -N1 -n1 -w "$n" rm -rf "$RAY_TMP" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

ENVA=()
while IFS='=' read -r k v; do [[ -n "$k" ]] && ENVA+=(--env "$k=$v"); done < <(nccl_env)

srun --overlap -N1 -n1 -w "$HEAD" apptainer exec $(apt_args) "${ENVA[@]}" "$VLLM_SIF" \
  $RAY_CLI start --head --node-ip-address="$HEAD_IP" --port=6379 \
  --temp-dir="$RAY_TMP" --num-gpus=8 --block >"$V/ray_head.log" 2>&1 &
for i in $(seq 1 60); do
  (echo >/dev/tcp/$HEAD_IP/6379) 2>/dev/null && break
  sleep 1
done
srun --overlap -N1 -n1 -w "$WORKER" apptainer exec $(apt_args) "${ENVA[@]}" "$VLLM_SIF" \
  $RAY_CLI start --address="$HEAD_IP:6379" --temp-dir="$RAY_TMP" --num-gpus=8 --block \
  >"$V/ray_worker.log" 2>&1 &

ngpu=0
for i in $(seq 1 90); do
  ngpu=$(apptainer exec $(apt_args) --env RAY_ADDRESS="$HEAD_IP:6379" "$VLLM_SIF" $PY_C -c "
import ray;ray.init(address='auto',logging_level='ERROR')
print(int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null | tail -1)
  [[ "${ngpu:-0}" -ge 16 ]] && break
  sleep 2
done
say "ray cluster GPUs: ${ngpu:-0}/16"
if [[ "${ngpu:-0}" -ge 16 ]]; then
  say "OK: 2-node ray cluster formed"
else
  say "FAIL: ray never reached 16 GPUs"
  tail -15 "$V/ray_head.log" "$V/ray_worker.log" 2>/dev/null | sed 's/^/    /' | tee -a "$V/STATE.txt"
  FAIL=1
fi

cleanup; trap - EXIT
if [[ $FAIL -eq 0 ]]; then
  say "VERIFY PASSED -- both prior failure modes are fixed"
  exit 0
fi
say "VERIFY FAILED"
exit 1
