#!/bin/bash
# Guard for a running Hyperloom session (runs on the AMD node, next to the run script).
#  - every CKPT_MIN minutes (default 30): copy the session's resume files to
#    <session dir>/../ckpt/<session name>/<UTC time>/ (state.json, storage/, reports/,
#    current_setting.sh, session_breakdown.json, manifest.json); keeps the newest KEEP (default 6).
#  - every 60 s: look for a Claude token / usage limit message in the run log and in every file
#    of the session dir changed since the last scan (Hyperloom catches the limit error itself and
#    writes it into critic-workdir/*/request.json, geak/*/llm_trace/*.jsonl, ...; it does not stop). On a hit: take a checkpoint, stop Hyperloom, the framework server and Ray,
#    and write STOPPED_TOKEN_LIMIT into the session dir. It never resumes by itself.
# Resume later (only when the user asks): RESUME=<session dir> bash run_step4.sh
#   [SESSION_DIR=<session dir>] guard.sh <run log> <state file>   (exits when the optimize process ends)
set -uo pipefail
LOG=${1:?run log}; ST=${2:?state file}
CKPT_MIN=${CKPT_MIN:-30}; KEEP=${KEEP:-6}
say() { echo "[$(date -Iseconds)] guard: $*" >>"$ST"; }
# Claude Code / Anthropic limit messages (usage limit, 5-hour / weekly limit, HTTP 429 rate limit)
RX='hit your [a-z0-9-]* ?limit|usage limit reached|limit reached.{0,20}resets|rate_limit_error|out of extra usage|credit balance is too low|weekly limit reached|Error code: 429'

sdir() { [[ -n ${SESSION_DIR:-} ]] && { echo "$SESSION_DIR"; return; }; grep -oE 'session_dir=[^ ]+' "$LOG" 2>/dev/null | tail -1 | cut -d= -f2; }
# the real optimizer process only (its cmdline starts with python); not the apptainer/bash wrappers
running() { pgrep -u "$(id -u)" -f '^(/[^ ]*/)?python3?[0-9.]* -m hyperloom.inference_optimizer.cli optimize' >/dev/null; }
# the run counts as ended only after 3 misses in a row (60 s apart)
alive() { local k; for k in 1 2 3; do running && return 0; [[ $k -lt 3 ]] && sleep 60; done; return 1; }

ckpt() {  # $1 = reason
  local S; S=$(sdir); [[ -n $S && -d $S ]] || return 0
  local C; C=$(dirname "$S")/ckpt/$(basename "$S")/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$C"
  for f in state.json session_breakdown.json current_setting.sh manifest.json; do
    [[ -f $S/$f ]] && cp -p "$S/$f" "$C/"; done
  [[ -d $S/reports ]] && cp -rp "$S/reports" "$C/"
  if [[ -d $S/storage ]]; then mkdir -p "$C/storage"
    for f in "$S"/storage/*; do
      if [[ $f == *.db ]] && command -v sqlite3 >/dev/null; then sqlite3 "$f" ".backup '$C/storage/$(basename "$f")'"
      else cp -p "$f" "$C/storage/"; fi
    done
  fi
  echo "$1" >"$C/REASON"
  ls -1d "$(dirname "$C")"/*/ 2>/dev/null | sort | head -n -"$KEEP" | xargs -r rm -rf
  say "checkpoint ($1) -> $C"
}

stop_all() {
  pkill -INT -u "$(id -u)" -f 'hyperloom.inference_optimizer.cli optimize'; sleep 60
  pkill -TERM -u "$(id -u)" -f 'hyperloom.inference_optimizer.cli optimize'; sleep 10
  pkill -TERM -u "$(id -u)" -f 'atom.entrypoints|vllm.entrypoints|sglang.launch_server'
  pkill -TERM -u "$(id -u)" -f 'raylet|gcs_server|ray::' ; sleep 5
  pkill -KILL -u "$(id -u)" -f 'hyperloom.inference_optimizer.cli optimize|atom.entrypoints|vllm.entrypoints|raylet|gcs_server'
}

say "start (checkpoint every ${CKPT_MIN} min, keep ${KEEP})"
for i in $(seq 1 ${WAIT_TRIES:-180}); do running && break; sleep 30; done   # up to 90 min for install
last=$(date +%s)
# only look at text written after this guard started (a resumed session keeps its old files)
# per-file byte offsets: only bytes written after the previous scan are read
M=$(dirname "$ST")/.guard-mark-$(hostname -s); touch "$M"
declare -A SZ
LOGOFF=$(( $(stat -c %s "$LOG" 2>/dev/null || echo 0) + 1 ))
# critic-workdir/ and robustness-workdir/ are skipped: their request files repeat recent event history,
# so an old limit message from the previous leg shows up again as "new" text (false stop, seen 3 times).
# Real limits are taken from new backend_error events in storage/coordinator.db (Hyperloom's own log).
SKIP=( -path "*/runtime" -o -path "*/critic-workdir" -o -path "*/robustness-workdir" )
DBSEQ=0
dbmax() { python3 -I -c 'import sqlite3,sys
try: print(sqlite3.connect("file:"+sys.argv[1]+"?mode=ro",uri=True).execute("select coalesce(max(seq),0) from events").fetchone()[0])
except Exception: print(0)' "$1/storage/coordinator.db"; }
dbhit() {  # new backend_error events after DBSEQ whose text is a limit message; prints the match
  python3 -I - "$1/storage/coordinator.db" "$DBSEQ" "$RX" <<'PY'
import sqlite3, sys, re
try:
    c = sqlite3.connect("file:" + sys.argv[1] + "?mode=ro", uri=True)
    for (p,) in c.execute("select payload from events where seq > ? and payload like '%backend_error%'", (int(sys.argv[2]),)):
        m = re.search(sys.argv[3], p, re.I)
        if m: print(m.group(0)); break
except Exception: pass
PY
}
seed() { local S; S=$(sdir); [[ -n $S && -d $S ]] || return 0   # only a resumed session exists at start
  DBSEQ=$(dbmax "$S")
  while IFS=' ' read -r sz f; do SZ[$f]=$sz; done < <(find "$S" \( "${SKIP[@]}" \) -prune -o -type f -printf '%s %p\n'); }
scan() {  # sets HIT from new run-log text, new bytes of changed session files, new backend_error events
  local S f sz T="$M.txt"; S=$(sdir); touch "$M.new"
  tail -c +"$LOGOFF" "$LOG" >"$T" 2>/dev/null; LOGOFF=$(( $(stat -c %s "$LOG" 2>/dev/null || echo 0) + 1 ))
  if [[ -n $S && -d $S ]]; then
    while IFS=' ' read -r sz f; do
      tail -c +$(( ${SZ[$f]:-0} + 1 )) "$f" 2>/dev/null | head -c 50000000 >>"$T"; SZ[$f]=$sz
    done < <(find "$S" \( "${SKIP[@]}" \) -prune -o -type f -newer "$M" -printf '%s %p\n')
  fi
  HIT=$(grep -oaiE "$RX" "$T" | head -1); rm -f "$T"
  [[ -z $HIT && -n $S && -f $S/storage/coordinator.db ]] && HIT=$(dbhit "$S")
  mv -f "$M.new" "$M"
}
seed
while alive; do
  sleep 60
  S=$(sdir)
  scan; hit=$HIT
  if [[ -n $hit ]]; then
    say "Claude limit message seen: '$hit' -> checkpoint and stop (no auto-resume)"
    ckpt "token-limit: $hit"
    [[ -n $S ]] && echo "$(date -Iseconds) $hit" >"$S/STOPPED_TOKEN_LIMIT"
    stop_all; say "stopped. Resume only when the user asks: RESUME=$S bash run_step4.sh"
    exit 3
  fi
  if (( $(date +%s) - last >= CKPT_MIN * 60 )); then ckpt periodic; last=$(date +%s); fi
done
ckpt "session ended"
say "optimize process ended; guard exits"
