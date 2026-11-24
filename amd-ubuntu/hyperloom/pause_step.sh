#!/bin/bash
# Pause the running chain step ON node6100 (same steps as guard.sh at a Claude limit):
# set AUTORESUME_STOP (no resume, chain stops), checkpoint, stop Hyperloom/ATOM/Ray, checkpoint again ("paused").
# The session dir then holds the paused state; the chain resumes it when started again.
#   bash pause_step.sh <session dir>
set -uo pipefail
H=/orcd/data/orcd/022/benchmarks/amd-ubuntu; L=$H/logs/hyperloom; S=${1:?session dir}
ST=$L/chain-20261117/chain.state
say() { echo "[$(date -Iseconds)] pause: $*" >>"$ST"; }
ckpt() {
  local C; C=$(dirname "$S")/ckpt/$(basename "$S")/$(date -u +%Y%m%dT%H%M%SZ)-$1; mkdir -p "$C"
  for f in state.json session_breakdown.json current_setting.sh manifest.json; do
    [[ -f $S/$f ]] && cp -p "$S/$f" "$C/"; done
  [[ -d $S/reports ]] && cp -rp "$S/reports" "$C/"
  if [[ -d $S/storage ]]; then mkdir -p "$C/storage"
    for f in "$S"/storage/*; do
      if [[ $f == *.db ]] && command -v sqlite3 >/dev/null; then sqlite3 "$f" ".backup '$C/storage/$(basename "$f")'"
      else cp -p "$f" "$C/storage/"; fi
    done
  fi
  echo "$1" >"$C/REASON"; say "checkpoint ($1) -> $C"
}
touch "$L/AUTORESUME_STOP"; say "AUTORESUME_STOP set"
ckpt before-pause
pkill -INT -u "$(id -u)" -f 'hyperloom.inference_optimizer.cli optimize'; sleep 60
pkill -TERM -u "$(id -u)" -f 'hyperloom.inference_optimizer.cli optimize'; sleep 10
pkill -TERM -u "$(id -u)" -f 'atom.entrypoints|vllm.entrypoints|sglang.launch_server'
pkill -TERM -u "$(id -u)" -f 'raylet|gcs_server|ray::'; sleep 5
pkill -KILL -u "$(id -u)" -f 'hyperloom.inference_optimizer.cli optimize|atom.entrypoints|vllm.entrypoints|raylet|gcs_server'
sleep 30
ckpt paused
say "stopped; running: $(pgrep -u "$(id -u)" -fc 'cli optimize|atom.entrypoints|run_step4|autoresume|run_chain')"
