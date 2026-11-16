#!/bin/bash
# Nov 12 rerun, candidates + profiles (notes-improve.md "How to fix each problem", parts 2 and 3).
# Runs ON node6101 with our own script atom/run_kimi_atom_mtp.sh (same flags and client as the
# step 3 ATOM baseline, image nightly_202610011450, no MTP). No Claude tokens are used.
#
#   Part 1, candidates, 3 repeats, interleaved (base, mega, ep, base, mega, ep, ...):
#     base  baseline flags (reference for this run)
#     mega  + --moe-backend mega   (MegaMoE: a8w4 = FP8 activations x FP4 expert weights)
#     ep    + --enable-expert-parallel
#     users 1, 8, 64, 128, 256; 1024 in / 1024 out; GSM8K in repeat 1 of each arm.
#     An arm whose server does not start is retried once (next repeat), then dropped.
#   Part 2, profiles of the baseline with rocprofv3 (kernel trace + stats), one server per user
#     level (1, 64, 256), a 5 s window that opens 12 min after start (after load, during the bench).
#   Part 3, analyze_candidates.py writes the results into hyperloom/rerun-report.md.
#
# Start from login:  ssh node6101 "nohup bash $H/hyperloom/run_candidates.sh >/dev/null 2>&1 </dev/null &"
# Cancel:            touch $H/logs/hyperloom/CANDIDATES_STOP   (checked between runs)
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks; H=$B/amd-ubuntu; L=$H/logs/hyperloom
REPS=${REPS:-3}; STOPF=$L/CANDIDATES_STOP
ST=$L/candidates-$(hostname -s).state; mkdir -p "$L" /dev/shm/shaohao-tmp
say() { echo "[$(date -Iseconds)] $*" >>"$ST"; }
TMPE="TMPDIR=/dev/shm/shaohao-tmp"
declare -A XARGS=( [base]="" [mega]="--moe-backend mega" [ep]="--enable-expert-parallel" )
declare -A FAILS=( [base]=0 [mega]=0 [ep]=0 )
RUNS=$L/candidates-runs-$(hostname -s).txt   # "<arm> <rep> <out dir>" per run, read by the analysis
rm -f "$STOPF"; : >"$RUNS"

newest_out() { ls -1dt "$H"/logs/$(hostname -s)/hyperloom-cand/atom/kimi_atom_mtp_"$1"_* 2>/dev/null | head -1; }

say "candidates start: reps=$REPS arms=base,mega,ep"
for rep in $(seq 1 "$REPS"); do
  for arm in base mega ep; do
    [[ -e $STOPF ]] && { say "CANDIDATES_STOP found -> stop"; exit 0; }
    (( FAILS[$arm] >= 2 )) && { say "$arm: failed twice, skipped"; continue; }
    g=0; (( rep == 1 || FAILS[$arm] == 1 )) && [[ ! -f $L/.cand-gsm8k-$arm ]] && g=1
    say "rep $rep arm $arm: start (gsm8k=$g)"
    RESULTS_SUBDIR=hyperloom-cand NO_REPORT=1 ARMS="0:1 8 64 128 256" OUTTAG="${arm}_r$rep" \
      EXTRA_ARGS="${XARGS[$arm]}" EXTRA_ENVS="$TMPE" GSM8K=$g \
      bash "$H/atom/run_kimi_atom_mtp.sh" >/dev/null 2>&1
    O=$(newest_out "${arm}_r$rep")
    if [[ -z $O || -f $O/k0/FAILED || ! -f $O/k0/c64.json ]]; then
      FAILS[$arm]=$(( FAILS[$arm] + 1 ))
      say "rep $rep arm $arm: FAILED (${FAILS[$arm]}x) dir=$O last server lines: $(tail -3 "$O/k0/atom_server.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
    else
      (( g == 1 )) && touch "$L/.cand-gsm8k-$arm"
      say "rep $rep arm $arm: done dir=$O"
    fi
    echo "$arm $rep ${O:-none}" >>"$RUNS"
  done
done
rm -f "$L"/.cand-gsm8k-*

# Part 2: rocprofv3 profiles of the baseline at 1, 64, 256 users
declare -A NP=( [1]=45 [64]=1280 [256]=1536 )
for C in 1 64 256; do
  [[ -e $STOPF ]] && { say "CANDIDATES_STOP found -> stop"; exit 0; }
  say "profile c=$C: start (rocprofv3 window 720 s + 5 s)"
  RESULTS_SUBDIR=hyperloom-cand NO_REPORT=1 ARMS="0:$C" OUTTAG="prof_c$C" NUM_PROMPTS=${NP[$C]} \
    EXTRA_ENVS="$TMPE" \
    PREFIX="rocprofv3 --kernel-trace --stats --summary --summary-output-file /out/rocprof_summary.txt -f csv -d /out/rocprof -P 720:5:1 --" \
    bash "$H/atom/run_kimi_atom_mtp.sh" >/dev/null 2>&1
  O=$(newest_out "prof_c$C")
  n=$(find "$O/k0/rocprof" -name '*kernel_stats.csv' 2>/dev/null | wc -l)
  say "profile c=$C: done dir=$O kernel_stats files=$n"
  echo "prof $C ${O:-none}" >>"$RUNS"
  # keep rank traces small: drop full kernel traces except one rank (the stats files stay)
  find "$O/k0/rocprof" -name '*kernel_trace.csv' 2>/dev/null | sort | tail -n +2 | xargs -r rm -f
done

say "analysis start"
python3 -I "$H/hyperloom/analyze_candidates.py" "$RUNS" >>"$ST" 2>&1; say "analysis rc=$?"
say "candidates DONE"
