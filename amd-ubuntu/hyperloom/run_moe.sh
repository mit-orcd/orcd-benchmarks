#!/bin/bash
# Nov 18: targeted MoE kernel run (hyperloom/status-chain-20261117.md, "worth continuing" step 1), then analysis.
# Runs ON node6100:  ssh node6100 "setsid nohup bash $H/hyperloom/run_moe.sh >/dev/null 2>&1 </dev/null &"
#  1. Hyperloom under run_step4_autoresume.sh (checkpoint every 30 min; at a Claude limit it stops, checks
#     every 60 min and resumes the same session): baseline ref_s3.sh (FP16 + step 3 setting), kernel agent
#     only (KERNEL=1 FRAMEWORK=0, --no-explore --no-warm-replay, kernel phase 75% of the time), 64 users,
#     target +25% so it does not stop early, 6 h budget. Non-limit failure: retried once, then stop.
#  2. If the kept change is env/flags only: our own A/B check (run_kimi_atom_mtp.sh, 64 users + GSM8K).
#  3. Facts file (python), then a headless Claude run writes results/kimi-k3-hyperloom-moe.md.
#     At a Claude limit: retry every 60 min (max 24). If it cannot run: a facts-only md is written.
# No push. State: logs/hyperloom/moe-20261118/state.   Cancel: touch logs/hyperloom/AUTORESUME_STOP
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks; H=$B/amd-ubuntu; SW=$B/amd-software; L=$H/logs/hyperloom
D=$L/moe-20261118; mkdir -p "$D"; ST=$D/state; OUTMD=$H/results/kimi-k3-hyperloom-moe.md
ENVF=$H/private/hyperloom/.env; CLAUDE_BIN=$HOME/.local/bin/claude
RX='hit your [a-z0-9-]* ?limit|usage limit reached|limit reached.{0,20}resets|rate_limit_error|out of extra usage|weekly limit reached|Error code: 429'
say() { echo "[$(date -Iseconds)] $*" >>"$ST"; }
exec 7>/dev/shm/shaohao-moe.lock; flock -n 7 || exit 0
stopped() { [[ -e $L/AUTORESUME_STOP ]] && { say "AUTORESUME_STOP -> stop"; exit 0; }; }
say "start"; stopped

# ---- 1. Hyperloom --------------------------------------------------------------------------------
ARST=$L/step4-autoresume-$(hostname -s).state
hl() {  # $1 = RESUME value (new | session dir), $2 = used hours
  local mark=$D/mark-$(date +%s); touch "$mark"
  RESUME=$1 USED_HOURS=$2 TOTAL_HOURS=6 CONC=64 CONCS=64 KERNEL=1 FRAMEWORK=0 REF=$H/hyperloom/ref_s3.sh \
    TARGET=25 EXTRA_OPT="--no-explore --no-warm-replay --phase-budget-kernel-pct 0.75" \
    bash "$H/hyperloom/run_step4_autoresume.sh" 7>&-
  local lg; lg=$(ls -1t "$L"/step4-A-run-*.log | head -1)
  [[ -z ${S:-} ]] && S=$(grep -oE 'session_dir=[^ ]+' "$lg" 2>/dev/null | tail -1 | cut -d= -f2)
  [[ -n ${S:-} ]] && echo "$S" >"$D/session"; rm -f "$mark"
}
S=$(cat "$D/session" 2>/dev/null)
if [[ ! -f $D/hl.done ]]; then
  hl "${S:-new}" "$(cat "$D/used" 2>/dev/null || echo 0)"; stopped
  sr=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('stop_reason') or '')" "$S/state.json" 2>/dev/null)
  last=$(tail -1 "$ARST")
  if [[ -z $S || ! -f $S/state.json ]] || { [[ -z $sr || $sr == signal || $sr == token_limit ]] && ! grep -qE 'budget of .* used up|resumes used|deadline' <<<"$last"; }; then
    say "Hyperloom failed (session=${S:-none}, stop_reason=${sr:-none}, last: $last) -> retry once"
    u=$(grep -oE 'used [0-9.]+ h' "$ARST" | tail -1 | grep -oE '[0-9.]+'); echo "${u:-0}" >"$D/used"
    hl "${S:-new}" "${u:-0}"; stopped
    sr=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('stop_reason') or '')" "$S/state.json" 2>/dev/null)
    [[ -n $S && -f $S/state.json ]] || { say "Hyperloom failed twice -> stop"; exit 1; }
  fi
  say "Hyperloom done: session $S stop_reason=${sr:-?}"; touch "$D/hl.done"
fi

# ---- 2. facts + own check ------------------------------------------------------------------------
python3 -I - "$S" "$D/facts.md" <<'EOF'
import json, sys, os, glob
S, out = sys.argv[1:3]
d = json.load(open(os.path.join(S, "state.json"))); cb = d.get("current_best") or {}
L = []
L.append("session: " + S)
for k in ("phase", "stop_reason", "baseline_tput", "cumulative_gain_validated", "conc", "target_summary"):
    L.append("%s: %s" % (k, d.get(k)))
for k in ("action", "variant_name", "tput", "tpot_mean_ms", "ttft_mean_ms", "extra_envs", "extra_server_args",
          "remove_args", "final_overlay", "workspace"):
    L.append("best.%s: %s" % (k, json.dumps(cb.get(k))[:1500]))
for e in cb.get("optimization_stack") or []:
    L.append("stack: " + json.dumps({k: e.get(k) for k in ("action", "operation_kind", "gain_pct", "accuracy",
             "hot_tput", "cold_tput", "candidate_extra_envs", "candidate_extra_server_args", "fingerprint")})[:1500])
kinds = {e.get("operation_kind") for e in (cb.get("optimization_stack") or [])[1:]}
L.append("kept_kinds: %s" % sorted(k for k in kinds if k))
open(out, "w").write("\n".join(L) + "\n")
print("env_only" if kinds and kinds <= {"env", "param"} and not cb.get("final_overlay") else ("none" if not kinds else "kernel"))
EOF
KEPT=$(python3 -I -c "
import json,sys;d=json.load(open(sys.argv[1]));cb=d.get('current_best') or {};st=(cb.get('optimization_stack') or [])[1:]
k={e.get('operation_kind') for e in st}
print('none' if not st else ('env_only' if k<= {'env','param'} and not cb.get('final_overlay') else 'kernel'))" "$S/state.json" 2>/dev/null)
say "kept change type: ${KEPT:-?}"
if [[ $KEPT == env_only && ! -f $D/check.done ]]; then
  read -r E A <<<"$(python3 -I -c "
import json,sys;cb=json.load(open(sys.argv[1]))['current_best']
print(' '.join('%s=%s'%kv for kv in (cb.get('extra_envs') or {}).items()).replace(' ','|') or '-', cb.get('extra_server_args') or '')" "$S/state.json")"
  if [[ $A == *online_quant_config* ]]; then say "own check skipped: kept flags are a full replace string, not a delta (Hyperloom numbers only)"; touch "$D/check.done"; fi
  say "own check: B envs '${E//|/ }' flags '$A'"
  [[ -f $D/check.done ]] || for tag in A B; do
    if [[ $tag == A ]]; then ee="ATOM_GDN_SSM_DTYPE=fp16 ATOM_PREFILL_DECODE_INTERVAL=4 ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000"; xa=""
    else ee="${E//|/ }"; xa="$A"; fi
    for t in 1 2; do
      ARMS="0:64" GSM8K=1 NO_REPORT=1 OUTTAG="check-moe-$tag" EXTRA_ENVS="$ee" EXTRA_ARGS="$xa" BATCHED=8192 \
        bash "$H/atom/run_kimi_atom_mtp.sh" >>"$D/check.out" 2>&1
      od=$(ls -1dt "$H"/logs/*/kimi-atom-mtp/atom/kimi_atom_mtp_check-moe-"$tag"_* | head -1)
      grep -E 'c=64 rc|GSM8K rc' "$od/STATE.txt" | sed "s/^/  $tag: /" >>"$ST"
      [[ -f $od/k0/c64.json && ! -f $od/k0/FAILED ]] && { echo "$tag: $od" >>"$D/check.dirs"; break; }
      say "own check $tag try $t failed"
    done
  done
  touch "$D/check.done"
fi

# ---- 3. analysis by Claude -----------------------------------------------------------------------
cat >"$D/prompt.txt" <<EOF
Write the file $OUTMD (Markdown). It summarizes a Hyperloom run that tried to make Kimi-K3 inference faster
with ATOM on 8 x AMD MI355X (TP8), by improving the MoE kernels.

Inputs (read only; do not run GPU jobs, do not change any other file):
- Facts: $D/facts.md and pipeline log $D/state
- Hyperloom session: $S (state.json, reports/, kernel-agent/, kernel-agent-workspace/, runs/, findings/, patches/).
  Look at what the kernel agent tried, which kernels it changed, what was kept or reverted, and why.
- Our own A/B check (only if it ran): directories listed in $D/check.dirs (STATE.txt, k0/c64.json, gsm8k/).
- Background: $H/hyperloom/status-chain-20261117.md (earlier steps; baseline of this run = FP16 + step 3 setting,
  1,852.8 output tok/s at 64 users in our own run, TPOT 32.86 ms, TTFT 823 ms, GSM8K strict 0.9591) and
  $H/plan-hyperloom.md (profile: MoE expert GEMMs 36.7%, all-reduce 21.7%, other GEMMs 19.5%, attention 7.6%).

Contents, in this order: setup (image, settings, budget, what was run); result table (baseline vs best:
output tok/s, TPOT = time per output token, TTFT = time to first token, GSM8K, when known); what was modified
(each kept change: kernel or file name, what it does); how much each change improved; why it improves
performance (the mechanism, e.g. less memory traffic, better tiling, fused steps), with evidence from the
session files; what was tried and not kept, and why; whether Hyperloom is worth running again for Kimi-K3.
If nothing was kept, say so plainly and explain why from the logs.
Style: short, simple sentences; precise technical terms, no figurative wording; numbers from the files only,
never guessed; say which numbers are Hyperloom's and which are ours; times in US Eastern time; paths relative
to $H. Never copy tokens, keys or the contents of any .env file.
EOF
run_claude() {
  ( set -a; . "$ENVF"; set +a
    cd "$H" && env -i HOME="$HOME" PATH=/usr/bin:/bin CLAUDE_CODE_OAUTH_TOKEN="$CLAUDE_CODE_OAUTH_TOKEN" \
      CLAUDE_CONFIG_DIR="$SW/cache/claude-analysis" \
      timeout 3600 "$CLAUDE_BIN" -p --model claude-opus-5-5 --max-turns 80 \
        --allowedTools "Read,Grep,Glob,Write,Bash(ls:*),Bash(python3:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(grep:*)" \
        --permission-mode acceptEdits --add-dir "$L" --add-dir "$H/results" --setting-sources "" \
        --strict-mcp-config --no-session-persistence <"$D/prompt.txt" ) >"$D/claude.out" 2>&1
}
mkdir -p "$SW/cache/claude-analysis"
for i in $(seq 1 24); do
  stopped
  m0=$(stat -c %Y "$OUTMD" 2>/dev/null || echo 0)
  run_claude; rc=$?
  if [[ $(stat -c %Y "$OUTMD" 2>/dev/null || echo 0) -gt $m0 ]]; then say "analysis written by Claude (rc=$rc) -> $OUTMD"; touch "$D/DONE"; say "DONE"; exit 0; fi
  if grep -qiE "$RX" "$D/claude.out"; then say "analysis: Claude limit, retry in 60 min ($i/24)"; sleep 3600
  else say "analysis: Claude rc=$rc, no file; $( (( i >= 2 )) && echo 'stop' || echo 'retry once')"; (( i >= 2 )) && break; sleep 60; fi
done
{ echo "# Kimi-K3 Hyperloom MoE kernel run (facts only)"; echo
  echo "Claude could not write the analysis (see \`logs/hyperloom/moe-20261118/state\`). Raw facts:"; echo
  echo '```'; sed "s|$H/||g" "$D/facts.md"; echo '```'; echo; echo "Pipeline log:"; echo; echo '```'
  sed "s|$H/||g" "$ST"; echo '```'; } >"$OUTMD"
say "facts-only md written -> $OUTMD"; touch "$D/DONE"; say "DONE"
