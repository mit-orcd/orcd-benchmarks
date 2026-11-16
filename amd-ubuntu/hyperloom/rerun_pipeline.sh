#!/bin/bash
# Nov 12 rerun pipeline (notes-improve.md "Order for the rerun"), runs ON node6100, all steps in order.
# Each step must pass before the next one starts; a failed step is retried once, then the pipeline
# stops and reports. Results go to hyperloom/rerun-report.md (part A).
#
#  T1  JSON quoting + temp folder (no tokens): Hyperloom's own code removes and re-adds a flag in our
#      reference flags (the path that broke the JSON), then an ATOM server is started with the result,
#      expanded unquoted like Hyperloom's launcher, TMPDIR on /dev/shm. Pass = server ready, one request
#      answered, no "invalid loads value", no Errno 39.
#  T2  helper file access (a few tokens): inside the Hyperloom container, `claude` must be on PATH
#      (so Hyperloom uses the CLI helpers, not the in-process fallback), and one `claude -p` with
#      Hyperloom's helper flags (bypassPermissions, --add-dir) must read /app/ATOM and the model config.
#  T3  1.5-hour Hyperloom test (tokens): new session, GEAK off, conc sweep 8,64. check_session.py must
#      pass: JSON ok, no Errno 39, explore numbers, helpers not blocked.
#  T4  fresh 12-hour session with auto-resume after Claude limits (tokens): run_step4_autoresume.sh
#      RESUME=new, GEAK off, conc sweep 1,8,64,128,256. Checked with check_session.py at the end.
#
# Start from login:  ssh node6100 "nohup bash $H/hyperloom/rerun_pipeline.sh >/dev/null 2>&1 </dev/null &"
# Cancel:            touch $H/logs/hyperloom/AUTORESUME_STOP   (checked between steps and by the loop)
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks; SW=$B/amd-software; W=$SW/hyperloom; H=$B/amd-ubuntu; L=$H/logs/hyperloom
SIF=$SW/sif/atom-dev-nightly_202610011450.sif; OV=$SW/cache/overlay/hyperloom-atom.img
MODEL=/scratch/Kimi-K3; LTMP=/dev/shm/shaohao-tmp; PORT=8042
ENVF=$H/private/hyperloom/.env; CLAUDE_BIN=$HOME/.local/bin/claude
STOPF=$L/AUTORESUME_STOP; LOCK=/dev/shm/shaohao-gpu.lock
T3_HOURS=${T3_HOURS:-1.5}; START_AT=${START_AT:-T1}   # START_AT=T3 skips T1/T2 (already passed)
ST=$L/rerun-pipeline-$(hostname -s).state; TD=$L/rerun-tests; mkdir -p "$TD" "$LTMP"
say() { echo "[$(date -Iseconds)] $*" >>"$ST"; }
rep() { python3 -I "$H/hyperloom/rerun_report.py" add A "$*" 2>>"$ST"; say "report: $*"; }
stop_requested() { [[ -e $STOPF ]] && { say "AUTORESUME_STOP found -> stop"; rep "Stopped by AUTORESUME_STOP."; exit 0; }; }
RX='hit your [a-z0-9-]* ?limit|usage limit|limit reached|rate_limit|429|weekly limit|out of extra usage|credit balance'

exec 7>/dev/shm/shaohao-rerun.lock
flock -n 7 || { say "another rerun pipeline is running, exit"; exit 1; }
rm -f "$STOPF"
say "pipeline start"
rep "Pipeline started on $(hostname -s)."

# ---------- one tiny Claude request: ok / limit / error (token never printed) ----------
probe() {
  local out rc
  out=$( set -a; . "$ENVF"; set +a
         env -i HOME="$HOME" PATH=/usr/bin:/bin CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN" \
           CLAUDE_CONFIG_DIR="$SW/cache/claude-probe" \
           timeout 120 "$CLAUDE_BIN" -p --model "${CLAUDE_MODEL:-claude-opus-5-5}" --tools "" \
             --system-prompt "Reply with the single word OK." --setting-sources "" --strict-mcp-config \
             --no-session-persistence "ping" 2>&1 ); rc=$?
  if (( rc == 0 )) && grep -qw OK <<<"$out"; then echo ok
  elif grep -qiE "$RX" <<<"$out"; then echo limit
  else echo "error(rc=$rc)"; fi
}
wait_claude() {  # wait (hourly checks) until Claude answers; max 7 days
  local r i
  for i in $(seq 1 168); do
    stop_requested
    r=$(probe); say "Claude check: $r"
    [[ $r == ok ]] && return 0
    [[ $i == 1 ]] && rep "Claude not available ($r); checking every 60 min."
    for _ in $(seq 1 60); do stop_requested; sleep 60; done
  done
  rep "Claude still not available after 7 days -> stop."; exit 1
}

# ---------- T1 ----------
t1() {
  local D=$TD/t1-$(date -u +%Y%m%dT%H%M%SZ) ARGS ready=0 i
  mkdir -p "$D"
  exec 9>"$LOCK"; flock 9
  ARGS=$(apptainer exec --cleanenv --overlay "$OV" -B /orcd/data/orcd/022 --pwd "$W" --env PYTHONPATH="$W" \
    "$SIF" python3 -I -c '
import sys; sys.path.insert(0, "'"$W"'")
from hyperloom.inference_optimizer.reference_script import parse_reference_script
from hyperloom.orchestrator.actions.executors._grid_server_args import (
    remove_server_args, merge_server_args, compact_json_server_args, validate_server_args_shell_safe)
r = parse_reference_script("'"$H"'/hyperloom/kimi_atom_reference.sh", framework="atom")
a = merge_server_args(remove_server_args(r.server_args, ["--max-num-seqs"]), "--max-num-seqs 64")
a = compact_json_server_args(a, "atom"); validate_server_args_shell_safe(a); print(a)' 2>"$D/args.err")
  echo "$ARGS" >"$D/args.txt"
  if ! grep -q '"global_quant_config"' <<<"$ARGS"; then
    say "T1: JSON lost its quotes after remove_server_args: $(cut -c1-300 <<<"$ARGS") $(tail -2 "$D/args.err")"
    flock -u 9; exec 9>&-; return 1
  fi
  say "T1: flags after Hyperloom's remove/merge keep the JSON quotes; starting ATOM"
  apptainer exec --cleanenv --overlay "$OV" -B /orcd/data/orcd/022 -B "$MODEL" \
    --env TMPDIR="$LTMP" --env EXTRA_ATOM_ARGS="$ARGS" --env HF_HOME="$SW/cache/hf" --env XDG_CACHE_HOME="$SW/cache/xdg" \
    --env AITER_LOG_LEVEL=WARNING --env NCCL_IB_DISABLE=1 "$SIF" \
    bash -c "exec python3 -m atom.entrypoints.openai_server --model $MODEL --server-port $PORT \$EXTRA_ATOM_ARGS" \
    >"$D/server.log" 2>&1 &
  local SP=$!
  for i in $(seq 1 240); do   # up to 40 min
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
    kill -0 $SP 2>/dev/null || break
    sleep 10
  done
  local ans=""
  if (( ready )); then
    ans=$(curl -s -m 300 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
      | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"].strip()[:60])' 2>/dev/null)
  fi
  pkill -TERM -u "$(id -u)" -f 'atom.entrypoints.openai_server' ; sleep 20
  pkill -KILL -u "$(id -u)" -f 'atom.entrypoints.openai_server|ModelRunner' 2>/dev/null; wait $SP 2>/dev/null
  rm -rf "${LTMP:?}"/* 2>/dev/null
  flock -u 9; exec 9>&-
  local le e39; le=$(grep -c "invalid loads value" "$D/server.log"); e39=$(grep -c "Errno 39" "$D/server.log")
  say "T1: ready=$ready answer='$ans' loads_err=$le errno39=$e39 (dir $D)"
  (( ready && le == 0 && e39 == 0 )) && [[ -n $ans ]] && { rep "**T1 passed** (JSON quoting + temp folder): Hyperloom's flag removal now keeps the JSON; ATOM started with TMPDIR on /dev/shm and answered (\"$ans\"). Folder \`logs/hyperloom/rerun-tests/$(basename "$D")\`."; return 0; }
  rep "T1 failed: ready=$ready, invalid-loads errors=$le, Errno 39=$e39. Last server lines: $(tail -3 "$D/server.log" | tr '\n' ' ' | cut -c1-300)"
  return 1
}

# ---------- T2 ----------
t2() {
  local D=$TD/t2-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$D" "$LTMP/t2cwd"
  local LN; LN=$(apptainer exec --cleanenv "$SIF" grep -n "moe_backend: str" /app/ATOM/atom/config.py | head -1 | cut -d: -f1)
  apptainer exec --cleanenv --overlay "$OV:ro" -B /orcd/data/orcd/022 -B "$MODEL" --pwd "$LTMP/t2cwd" \
    --env REPO_ROOT="$W" --env ENVF="$ENVF" "$SIF" bash -c '
      set -a; . "$ENVF"; set +a
      export PATH="$HOME/.local/bin:$PATH"
      echo "which=$(python3 -c "import shutil;print(shutil.which(\"claude\") or \"MISSING\")")"
      timeout 600 claude --print --output-format text --permission-mode bypassPermissions \
        --add-dir /app/ATOM/atom/ --add-dir /app/aiter-test/aiter/ --add-dir /opt/rocm/ \
        --model "${CLAUDE_MODEL:-claude-opus-5-5}" \
        "Use the Read tool to read /scratch/Kimi-K3/config.json and /app/ATOM/atom/config.py. Reply with exactly two lines and nothing else: line 1 = the value of model_type in config.json; line 2 = the line number in config.py of the line that starts with \"moe_backend: str\" (after leading spaces). If a read is refused, write REFUSED and the error text." 2>&1 | tail -20
    ' >"$D/out.txt" 2>&1
  local which; which=$(grep -m1 '^which=' "$D/out.txt" | cut -d= -f2)
  local mt ln; mt=$(grep -c "kimi_k3" "$D/out.txt"); ln=$(grep -cx "[[:space:]]*$LN[[:space:]]*" "$D/out.txt")
  say "T2: claude=$which model_type_found=$mt line_found=$ln (expected line $LN) refused=$(grep -c REFUSED "$D/out.txt")"
  if [[ $which != MISSING && $mt -ge 1 && $ln -ge 1 ]] && ! grep -q REFUSED "$D/out.txt"; then
    rep "**T2 passed** (helper file access): \`claude\` found at \`$which\` inside the container, so Hyperloom will use the CLI helpers with \`--add-dir\` and \`bypassPermissions\`. A test helper read \`/scratch/Kimi-K3/config.json\` (model_type kimi_k3) and \`/app/ATOM/atom/config.py\` (line $LN)."
    return 0
  fi
  rep "T2 failed: claude=$which, reply: $(tr '\n' ' ' <"$D/out.txt" | cut -c1-300)"
  return 1
}

# ---------- T3 / T4 helpers ----------
newest_new_log() { ls -1t "$L"/step4-A-run-new-*.log 2>/dev/null | head -1; }
sdir_of() { grep -oE 'session_dir=[^ ]+' "$1" 2>/dev/null | tail -1 | cut -d= -f2; }

t3() {
  local LOGF S res
  until flock -n "$LOCK" true 2>/dev/null; do say "T3: GPU lock busy, wait 10 min"; sleep 600; stop_requested; done
  say "T3: Hyperloom test, new session, ${T3_HOURS} h, GEAK off, framework agent off, conc sweep 8,64"
  RESUME="" HOURS="$T3_HOURS" KERNEL=0 FRAMEWORK=0 CONCS=8,64 bash "$H/hyperloom/run_step4.sh"
  LOGF=$(newest_new_log); S=$(sdir_of "$LOGF")
  [[ -n $S && -d $S ]] || { rep "T3 failed: no session folder found (log \`$(basename "$LOGF")\`)."; return 1; }
  if [[ -f $S/STOPPED_TOKEN_LIMIT ]]; then
    rep "T3 was stopped by a Claude limit (session \`$(basename "$S")\`); it will run again when Claude is available."
    return 2
  fi
  res=$(python3 -I "$H/hyperloom/check_session.py" "$S" "$LOGF" 2>&1)
  echo "$res" >"$TD/t3-check-$(basename "$S").txt"
  rep "T3 check of session \`Kimi-K3/$(basename "$S")\` (log \`$(basename "$LOGF")\`):"
  while IFS= read -r line; do [[ $line == RESULT* ]] || python3 -I "$H/hyperloom/rerun_report.py" add A "  $line" 2>>"$ST"; done <<<"$res"
  if grep -q '^RESULT PASS' <<<"$res"; then rep "**T3 passed.**"; return 0; fi
  rep "T3 failed: $(grep '^RESULT' <<<"$res")"; return 1
}

run_step() {  # $1 = name, $2 = function; retry once on failure; limit stops (rc 2) wait and rerun
  local n=0 rc lim=0
  while :; do
    stop_requested
    "$2"; rc=$?
    (( rc == 0 )) && return 0
    if (( rc == 2 && lim < 3 )); then lim=$((lim + 1)); wait_claude; continue; fi
    n=$((n + 1))
    if (( n >= 2 )); then rep "$1 failed twice -> pipeline stops here (retry-once rule)."; say "$1 failed twice, stop"; exit 1; fi
    rep "$1 failed once -> retry."
  done
}

if [[ $START_AT == T1 ]]; then
  run_step T1 t1
  wait_claude
  run_step T2 t2
fi
wait_claude
run_step T3 t3

# ---------- T4 ----------
stop_requested
rep "T4: fresh 12-hour session started with auto-resume after Claude limits (GEAK off, framework agent off, conc sweep 1,8,64,128,256). Progress: \`logs/hyperloom/step4-autoresume-node6100.state\`."
say "T4: start autoresume"
ARST=$L/step4-autoresume-$(hostname -s).state
N0=$(wc -l <"$ARST" 2>/dev/null || echo 0)   # session A's old legs are above this line
RESUME=new TOTAL_HOURS=12 KERNEL=0 FRAMEWORK=0 CONCS=1,8,64,128,256 bash "$H/hyperloom/run_step4_autoresume.sh"
say "T4: autoresume ended rc=$?"
LOGF=$(newest_new_log); S=$(sdir_of "$LOGF")
# the 12 h session's log may be a resume log; take the session from the autoresume state
S2=$(tail -n +"$((N0 + 1))" "$ARST" | grep -oE 'leg [0-9]+: /[^,]+' | tail -1 | awk '{print $3}')
[[ -n $S2 && -d $S2 ]] && S=$S2
if [[ -n $S && -d $S ]]; then
  res=$(python3 -I "$H/hyperloom/check_session.py" "$S" "$LOGF" 2>&1)
  echo "$res" >"$TD/t4-check-$(basename "$S").txt"
  rep "T4 ended. Check of session \`Kimi-K3/$(basename "$S")\`:"
  while IFS= read -r line; do [[ $line == RESULT* ]] || python3 -I "$H/hyperloom/rerun_report.py" add A "  $line" 2>>"$ST"; done <<<"$res"
  rep "T4 $(grep '^RESULT' <<<"$res"). Last auto-resume lines: $(tail -2 "$L/step4-autoresume-$(hostname -s).state" | tr '\n' ' ' | cut -c1-300)"
else
  rep "T4 ended, but no session folder was found. See \`logs/hyperloom/step4-autoresume-node6100.state\`."
fi
say "pipeline DONE"
rep "Pipeline finished."
