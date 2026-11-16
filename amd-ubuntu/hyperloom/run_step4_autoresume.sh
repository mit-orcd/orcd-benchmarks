#!/bin/bash
# Hyperloom step 4 session A WITH automatic resume after Claude limits, until step 4 is completed.
# (run_step4.sh alone = stop at a limit, no auto-resume. This loop calls it unchanged.)
#
# Loop (each pass = one "leg"):
#   1. Before every leg, check every CHECK_MIN (60) minutes with one tiny Claude request (same token
#      and model as Hyperloom, no tools) until Claude answers (= limit over, quota left).
#   2. Run the leg: RESUME=<session dir> run_step4.sh (= Hyperloom --resume-from). The first leg
#      resumes the newest session (default: the last run), so earlier work is kept.
#      guard.sh: checkpoint every 30 min; at a Claude limit it checkpoints, stops everything and
#      writes STOPPED_TOKEN_LIMIT in the session dir.
#   3. If the leg stopped at a limit: write stop_reason=token_limit into state.json (so Hyperloom does
#      not count the waiting time against its budget), then go back to 1 with the remaining hours.
#   4. If the leg ended any other way, step 4 is completed (or failed for another reason): stop.
# Safety limits: MAX_RESUMES (20), DEADLINE_HOURS (168 = 7 days), file logs/hyperloom/AUTORESUME_STOP
# (touch it to cancel; checked every minute while waiting).
#
# Start on node6100 from login:
#   ssh node6100 "nohup bash $H/hyperloom/run_step4_autoresume.sh >/dev/null 2>&1 </dev/null &"
# Settings: RESUME (newest session dir; "new" = start a new session), TOTAL_HOURS (12),
#   USED_HOURS (1.75 = Claude-active time of the last run before its limit), CHECK_MIN (60)
# The token is read from .env only inside the check's subshell and is never printed.
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks; H=$B/amd-ubuntu; L=$H/logs/hyperloom; SW=$B/amd-software
CHECK_MIN=${CHECK_MIN:-60}; MAX_RESUMES=${MAX_RESUMES:-20}; DEADLINE_HOURS=${DEADLINE_HOURS:-168}
TOTAL_HOURS=${TOTAL_HOURS:-12}; USED_HOURS=${USED_HOURS:-1.75}
STOPF=$L/AUTORESUME_STOP; LOCK=/dev/shm/shaohao-gpu.lock
ENVF=$H/private/hyperloom/.env; CLAUDE_BIN=${CLAUDE_BIN:-$HOME/.local/bin/claude}
ST=$L/step4-autoresume-$(hostname -s).state
say() { echo "[$(date -Iseconds)] autoresume: $*" >>"$ST"; }
END=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))
RX='hit your [a-z0-9-]* ?limit|usage limit|limit reached|rate_limit|429|weekly limit|out of extra usage|credit balance'

# default: resume the newest session that has a state.json
if [[ -z ${RESUME:-} ]]; then
  RESUME=$(ls -1dt "$L"/Kimi-K3/*/state.json 2>/dev/null | grep -v /ckpt/ | head -1 | xargs -r dirname)
elif [[ $RESUME == new ]]; then RESUME=""; USED_HOURS=0; fi

# never two copies of this loop
exec 8>/dev/shm/shaohao-autoresume.lock
flock -n 8 || { say "another auto-resume loop is running, exit"; exit 1; }
rm -f "$STOPF"

newest_log() { ls -1t "$L"/step4-A-run*.log 2>/dev/null | head -1; }
sdir_of() { grep -oE 'session_dir=[^ ]+' "$1" 2>/dev/null | tail -1 | cut -d= -f2; }

# one tiny Claude request; prints ok / limit / error (never the token or the reply text)
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

waitfor() {  # $1 = seconds; returns 1 if cancelled or past the deadline
  local until=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < until )); do
    [[ -e $STOPF ]] && { say "AUTORESUME_STOP found -> cancel"; return 1; }
    (( $(date +%s) > END )) && { say "deadline (${DEADLINE_HOURS} h) passed -> cancel"; return 1; }
    sleep 60
  done
}

# mark a guard stop in state.json, so the resume re-anchors the budget (waiting time not counted)
mark_stop() {  # $1 = session dir
  cp -p "$1/state.json" "$1/state.json.before-autoresume-$(date -u +%Y%m%dT%H%M%SZ)"
  python3 -I - "$1/state.json" <<'EOF'
import json, os, sys, datetime
p = sys.argv[1]; d = json.load(open(p))
if not d.get("stop_reason"):
    d["stop_reason"] = "token_limit"
    d["stop_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t = p + ".tmp"; json.dump(d, open(t, "w"), indent=2); os.replace(t, p)
    print("stop_reason set to token_limit")
else:
    print("stop_reason already", d["stop_reason"])
EOF
}

mkdir -p "$SW/cache/claude-probe"
say "start: resume=${RESUME:-new session} total ${TOTAL_HOURS} h, used ${USED_HOURS} h, check every $CHECK_MIN min, max resumes $MAX_RESUMES, deadline ${DEADLINE_HOURS} h"
n=0
while :; do
  # 1. wait until Claude answers (limit over, quota left)
  while :; do
    [[ -e $STOPF ]] && { say "AUTORESUME_STOP found -> cancel"; exit 0; }
    r=$(probe); say "limit check: $r"
    [[ $r == ok ]] && break
    waitfor $(( CHECK_MIN * 60 )) || exit 0
  done
  # wait for the GPU lock to be free (another GPU job may be running on this node)
  until flock -n "$LOCK" true 2>/dev/null; do say "GPU lock busy, check again in 10 min"; waitfor 600 || exit 0; done
  # 2. run one leg with the hours that are left
  LEFT=$(python3 -c "print(max(1.0, round($TOTAL_HOURS - $USED_HOURS, 2)))")
  say "leg $n: ${RESUME:-new session}, budget $LEFT h"
  t0=$(date +%s)
  RESUME="$RESUME" HOURS="$LEFT" bash "$H/hyperloom/run_step4.sh" 8>&-
  USED_HOURS=$(python3 -c "print(round($USED_HOURS + ($(date +%s) - $t0) / 3600, 2))")
  LOGF=$(newest_log); S=${RESUME:-$(sdir_of "$LOGF")}
  # 4. ended without a limit stop -> done
  if [[ -z $S || ! -f $S/STOPPED_TOKEN_LIMIT ]]; then
    sr=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('stop_reason',''))" "$S/state.json" 2>/dev/null)
    say "leg $n ended without a Claude limit stop (stop_reason=${sr:-?}, used ${USED_HOURS} h) -> step 4 done, no resume"; exit 0
  fi
  # 3. stopped at a limit -> mark it and loop
  say "leg $n stopped at a Claude limit (used ${USED_HOURS} h): $(mark_stop "$S")"
  n=$(( n + 1 ))
  if (( n > MAX_RESUMES )); then say "$MAX_RESUMES resumes used -> stop"; exit 0; fi
  if python3 -c "import sys; sys.exit(0 if $USED_HOURS < $TOTAL_HOURS - 0.25 else 1)"; then :; else
    say "budget of $TOTAL_HOURS h used up -> stop"; exit 0; fi
  RESUME=$S
  waitfor $(( CHECK_MIN * 60 )) || exit 0
done
