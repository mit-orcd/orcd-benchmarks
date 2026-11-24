#!/bin/bash
# Nov 17 chain: plan-hyperloom.md §11 steps 2-5, one after another, on node6100.
# Step 1 (image nightly_202610011450 + ATOM_GDN_SSM_DTYPE=fp16) is the baseline of every step (ref_fp16*.sh).
#   s2  server settings only, target 256 users (sweep 128,256), 4 h      KERNEL=0 FRAMEWORK=0
#   s3  kernel agent (MoE), target 64 users (sweep 64,256), 12 h         KERNEL=1 FRAMEWORK=0
#   s4  framework agent (all-reduce etc.), may patch /app/ATOM, 8 h      KERNEL=0 FRAMEWORK=1
#   s5  low load with DSpark speculative decoding, target 1 user (sweep 1,8), 4 h
# Each step runs under run_step4_autoresume.sh: guard.sh checkpoints every 30 min; at a Claude limit
# it checkpoints and stops, then the loop checks Claude every 60 min and resumes the same session.
# Restart: this script keeps its own state in logs/hyperloom/chain-20261117/ (<step>.session, .used,
# .tries, .done). Started again (by hand or by chain_watchdog.sh from cron), it skips done steps and
# resumes the running step's session. Another failure than a Claude limit: retried once, then the step
# is marked FAILED and the chain goes on. Cancel everything: touch logs/hyperloom/AUTORESUME_STOP
#   Start from login: ssh node6100 "nohup bash $H/hyperloom/run_chain.sh >/dev/null 2>&1 </dev/null &"
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks; H=$B/amd-ubuntu; L=$H/logs/hyperloom; C=$L/chain-20261117
STOPF=$L/AUTORESUME_STOP; REP=$H/hyperloom/status-chain-20261117.md
mkdir -p "$C"; ST=$C/chain.state
say() { echo "[$(date -Iseconds)] $*" >>"$ST"; }

exec 7>/dev/shm/shaohao-chain.lock
flock -n 7 || exit 0
[[ -e $STOPF ]] && { say "AUTORESUME_STOP present -> not starting"; exit 0; }
# an auto-resume loop left over from a killed chain: wait for it to end first
while ! flock -n /dev/shm/shaohao-autoresume.lock true 2>/dev/null; do
  say "an auto-resume loop is still running, wait 10 min"; sleep 600
  [[ -e $STOPF ]] && exit 0
done

[[ -f $REP ]] || cat >"$REP" <<'EOF'
# Hyperloom chain of Nov 17 (plan-hyperloom.md §11), results per step

Written by `hyperloom/run_chain.sh` when each step ends. Baseline of every step: image
`nightly_202610011450` + `ATOM_GDN_SSM_DTYPE=fp16`. Output tok/s are totals for all 8 GPUs.
Progress: `logs/hyperloom/chain-20261117/chain.state`.

EOF

STEPS=(
  "s2|256|128,256|0|0|4|ref_fp16.sh||server settings at 128-256 users"
  "s3|64|64,256|1|0|12|ref_fp16.sh||kernel agent on the MoE"
  "s4|64|64,256|0|1|8|ref_fp16.sh|/app/ATOM|framework agent (communication)"
  "s5|1|1,8|0|0|4|ref_fp16_mtp.sh||low load with DSpark speculative decoding"
)

summary() {  # $1 step, $2 session dir, $3 what, $4 status
  python3 -I - "$1" "$2" "$3" "$4" >>"$REP" <<'EOF'
import json, sys, os
step, sdir, what, status = sys.argv[1:5]
print(f"## {step}: {what}\n")
print(f"- Status: {status}. Session: `{os.path.relpath(sdir, '/orcd/data/orcd/022/benchmarks/amd-ubuntu') if sdir else 'none'}`")
try:
    d = json.load(open(os.path.join(sdir, "state.json")))
except Exception as e:
    print(f"- No state.json ({e.__class__.__name__})\n"); sys.exit()
b = d.get("baseline_tput"); g = d.get("cumulative_gain_validated")
print(f"- Stop reason: {d.get('stop_reason') or '-'}; target users: {d.get('conc')}")
print(f"- Baseline: {b:,.1f} tok/s" if isinstance(b, (int, float)) else "- Baseline: -")
print(f"- Validated gain: {g:+.2f}%" if isinstance(g, (int, float)) else "- Validated gain: -")
cb = d.get("current_best") or {}
if cb:
    print(f"- Best setting ({cb.get('action', '?')}): `{(cb.get('effective_extra_server_args') or '')[:400]}`")
    envs = cb.get("effective_env") or cb.get("env") or {}
    if envs: print(f"- Best envs: `{json.dumps(envs)[:300]}`")
for f in ("reports/final.md",):
    if os.path.exists(os.path.join(sdir, f)): print(f"- Report: `{os.path.relpath(os.path.join(sdir, f), '/orcd/data/orcd/022/benchmarks/amd-ubuntu')}`")
print()
EOF
}

say "chain start"
for row in "${STEPS[@]}"; do
  IFS='|' read -r name conc concs k f hrs ref fp what <<<"$row"
  [[ -f $C/$name.done ]] && continue
  while :; do
    [[ -e $STOPF ]] && { say "AUTORESUME_STOP found -> chain stops before/after $name"; exit 0; }
    S=$(cat "$C/$name.session" 2>/dev/null); used=$(cat "$C/$name.used" 2>/dev/null || echo 0)
    tries=$(cat "$C/$name.tries" 2>/dev/null || echo 0)
    mark=$C/$name.start-$(date +%s); touch "$mark"
    ARST=$L/step4-autoresume-$(hostname -s).state; n0=$(( $(wc -l <"$ARST" 2>/dev/null || echo 0) + 1 ))
    say "$name ($what): start, session=${S:-new}, used ${used} h of $hrs h, try $((tries + 1))"
    RESUME=${S:-new} USED_HOURS=$used TOTAL_HOURS=$hrs CONC=$conc CONCS=$concs KERNEL=$k FRAMEWORK=$f \
      REF=$H/hyperloom/$ref FRAMEWORK_PATH=$fp \
      bash "$H/hyperloom/run_step4_autoresume.sh" 7>&- &
    AP=$!
    while kill -0 "$AP" 2>/dev/null; do
      if [[ -z $S ]]; then
        lg=$(find "$L" -maxdepth 1 -name 'step4-A-run-*.log' -newer "$mark" 2>/dev/null | head -1)
        s=$(grep -oE 'session_dir=[^ ]+' "$lg" 2>/dev/null | tail -1 | cut -d= -f2)
        [[ -n $s ]] && { S=$s; echo "$S" >"$C/$name.session"; say "$name: session $S"; }
      fi
      # Claude-active hours so far (from the auto-resume log of this step), for a restart
      u=$(tail -n +"$n0" "$ARST" 2>/dev/null | grep -oE 'used [0-9.]+ h' | tail -1 | grep -oE '[0-9.]+')
      [[ -n $u ]] && echo "$u" >"$C/$name.used"
      sleep 60
    done
    wait "$AP"
    rm -f "$mark"
    [[ -e $STOPF ]] && { say "$name: AUTORESUME_STOP found -> chain stops"; exit 0; }
    lg=$(ls -1t "$L"/step4-A-run-*.log 2>/dev/null | head -1)
    [[ -z $S ]] && { S=$(grep -oE 'session_dir=[^ ]+' "$lg" 2>/dev/null | tail -1 | cut -d= -f2); [[ -n $S ]] && echo "$S" >"$C/$name.session"; }
    # Hyperloom always exits rc=1; judge by its stop_reason and by how the auto-resume loop ended
    orc=$(grep -oE 'optimize rc=[0-9]+' "$lg" 2>/dev/null | tail -1 | cut -d= -f2)
    arl=$(tail -n +"$n0" "$ARST" 2>/dev/null | tail -1)
    sr=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('stop_reason') or '')" "$S/state.json" 2>/dev/null)
    if [[ -n $S && -f $S/state.json ]] && { [[ -n $sr && $sr != signal && $sr != token_limit ]] \
         || grep -qE 'budget of .* used up|resumes used|deadline' <<<"$arl"; }; then
      say "$name: finished (optimize rc=${orc:-?}, stop_reason=${sr:--})"
      summary "$name" "$S" "$what" "finished"; touch "$C/$name.done"; break
    fi
    tries=$((tries + 1)); echo "$tries" >"$C/$name.tries"
    say "$name: FAILED try $tries (optimize rc=${orc:-?}, session=${S:-none}, log $lg): $(grep -m1 -E 'Error|ERROR' "$lg" 2>/dev/null | cut -c1-250)"
    if (( tries >= 2 )); then
      summary "$name" "$S" "$what" "FAILED twice (log \`${lg#$H/}\`)"; touch "$C/$name.done"; break
    fi
    sleep 300
  done
done
touch "$C/ALL_DONE"; say "chain DONE (all steps)"
