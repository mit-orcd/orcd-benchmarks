#!/usr/bin/env bash
# Tear down the server + Ray cluster this job created. Never touches anything else.
#
# Usage: ./stop-vllm-server.sh <server_outdir>
set -uo pipefail
cd "$(dirname "$0")"
source common/env.sh

OUT="${1:-$(cat "$LOG_ROOT/CURRENT_SERVER_DIR.txt" 2>/dev/null)}"
[[ -n "$OUT" && -d "$OUT" ]] || { echo "usage: $0 <server_outdir>" >&2; exit 1; }
say() { echo "[$(date -Iseconds)] stop: $*"; }

module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null

# Only ever the PID this job recorded. No pkill, no pattern matching on process names:
# on a shared cluster that is how you kill someone else's inference server, and it is
# the specific behaviour the ATOM wrappers were written to avoid.
if [[ -f "$OUT/server.pid" ]]; then
  PID=$(cat "$OUT/server.pid")
  if kill -0 "$PID" 2>/dev/null; then
    say "TERM $PID"
    kill -TERM "$PID" 2>/dev/null
    for i in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$PID" 2>/dev/null && { say "still alive, KILL $PID"; kill -KILL "$PID" 2>/dev/null; }
  else
    say "server pid $PID already gone"
  fi
fi

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  for n in "${NODES[@]}"; do
    say "ray stop on $n"
    srun --overlap -N1 -n1 -w "$n" \
      apptainer exec $(apt_args) "$VLLM_SIF" $RAY_CLI stop --force >/dev/null 2>&1 || true
    srun --overlap -N1 -n1 -w "$n" rm -rf "/dev/shm/ray-${SLURM_JOB_ID:-none}" >/dev/null 2>&1 || true
  done
  say "GPU state after teardown:"
  srun --overlap -N1 -n1 -w "${NODES[0]}" \
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/    /'
fi
say "done"
