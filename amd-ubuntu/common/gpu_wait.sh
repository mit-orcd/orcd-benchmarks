#!/usr/bin/env bash
# Source after common/env.sh. Helpers for jobs on the ssh-only AMD nodes (not in Slurm):
#   wait_for_others   block until no other user's process is on the GPUs (2 clear checks 60 s apart)
#   take_gpu_lock     block until our own previous jobs on this node release the node GPU lock
# Both log through say() if the caller defines it.
_gw_say() { if declare -F say >/dev/null; then say "$@"; else echo "[$(date -Iseconds)] $*"; fi; }

# Busy = a KFD (GPU) process owned by someone else, or GPU utilisation with no process of ours.
others_on_gpu() {
  local me pids p u mine=0 other=""
  me=$(id -un)
  pids=$(rocm-smi --showpids 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')
  for p in $pids; do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [[ -z $u ]] && continue
    if [[ $u == "$me" ]]; then mine=1; else other+="$u:$p "; fi
  done
  [[ -n $other ]] && { echo "$other"; return 0; }
  if (( ! mine )); then
    local busy; busy=$(rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print $NF}' | grep -cv '^0$')
    (( busy > 0 )) && { echo "$busy GPU(s) busy, owner unknown"; return 0; }
  fi
  return 1
}

wait_for_others() {
  local clear=0 w
  while (( clear < 2 )); do
    if w=$(others_on_gpu); then
      _gw_say "HOLD: other users on the GPUs ($w); re-check in 5 min"; clear=0; sleep 300
    else
      clear=$((clear + 1)); (( clear < 2 )) && sleep 60
    fi
  done
  _gw_say "GPUs free of other users"
}

# fd 9 holds the lock until the caller exits (or runs: flock -u 9; exec 9>&-).
take_gpu_lock() {
  exec 9>/dev/shm/shaohao-gpu.lock
  _gw_say "waiting for the node GPU lock (our earlier jobs on this node)"
  flock 9
  _gw_say "got the node GPU lock"
}
