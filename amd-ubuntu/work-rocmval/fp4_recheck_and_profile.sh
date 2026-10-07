#!/usr/bin/env bash
# fp4 recheck (2026-10-07): is the ROCm 7.14 N=8 fp4 drop real, and if so, why?
#
# Stage 1, confirm. RVS gst fp4 with the sweep's own confs (same matrix, 30 s, log every 3 s),
# 3 repeats, on BOTH stacks (host ROCm 7.2.4 and ROCm 7.14):
#   n1      1 GPU                               (1x_fp4.conf)
#   n2, n4  2 / 4 GPUs from one rvs process     (2x/4x_fp4.conf)  -- where does the drop start?
#   n8      8 GPUs from one rvs process         (8x_fp4.conf, the standard run)
#   m8      8 rvs processes, one GPU each       (8x conf with one device per process)
# Every run samples clocks/power (rocm-smi, 2 s); single-process runs also sample per-thread
# CPU of rvs (pidstat -t, 2 s).
# Confirmed = ROCm 7.14 n8 per-GPU mean < 0.85 x 7.14 n1, while 7.14 m8 >= 0.90 x 7.14 n1.
#
# Stage 2, profile (only if confirmed). 10 s runs of n1 / n8 / m8 on 7.14 and n8 on 7.2.4 under
# rocprofv3 --kernel-trace: per GPU the GEMM kernel name, kernel duration and the idle gap
# between kernels. Longer kernels => device side (clocks, a different/slower kernel); same
# kernels with big gaps => host side (launch/synchronisation in the one process). Also a 10 s
# `perf record -g` of the 7.14 n8 rvs process (host hot spots), if perf is allowed.
#
# Waits for other users and for our running jobs on the node (GPU lock), then runs.
#   ./launch.sh node6101 work-rocmval/fp4_recheck_and_profile.sh
# Output: logs/<node>/rvs/fp4_recheck_<ts>/ ; report results/vs-amd-cloud/rvs-fp4-recheck.md
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks/amd-ubuntu
source $B/common/env.sh
source $B/common/gpu_wait.sh
H=$(hostname -s); TS=$(date +%Y%m%d_%H%M%S)
OUT=$B/logs/$H/rvs/fp4_recheck_$TS; mkdir -p "$OUT"
say() { echo "[$(date -Iseconds)] $H $*" | tee -a "$OUT/STATE.txt"; }
REPS=${REPS:-3}
SWEEP=$(ls -d $B/logs/$H/rvs/sweep_* | head -n 1)
say "fp4 recheck on $H, confs from $SWEEP, $REPS repeats"

# our Kimi job on this node first (it may not hold the lock yet while it loads), then the lock
while pgrep -u "$(id -un)" -f "atom/run_kimi_[a]mdrecipe" >/dev/null; do
  say "our Kimi job still running on $H; re-check in 10 min"; sleep 600
done
take_gpu_lock
wait_for_others

sample_smi() { while :; do echo "## $(date +%s)"; rocm-smi --showpower --showclocks --showtemp 2>/dev/null | grep -E "Power|sclk|mclk|junction"; sleep 2; done; }

run_rvs() {   # dir stack conf [mode=single|multi] [prefix cmd...]  -> runs rvs, logs in dir
  local d=$1 stack=$2 conf=$3 mode=${4:-single}; shift 4 || shift $#
  local pre=("$@")
  mkdir -p "$d"
  ( export ROCM_STACK=$stack; source $B/common/env.sh >/dev/null 2>&1
    echo "RVS_BIN=$RVS_BIN ROCM_STACK=$stack" >"$d/info.txt"
    sample_smi >"$d/smi.txt" 2>&1 & sp=$!
    if [[ $mode == single ]]; then
      cp "$conf" "$d/run.conf"
      "${pre[@]}" "$RVS_BIN" -c "$d/run.conf" >"$d/rvs.log" 2>&1 &
      rp=$!
      sleep 3; pidstat -t -u -p "$(pgrep -P $rp -n || echo $rp)" 2 >"$d/pidstat.txt" 2>&1 & pp=$!
      wait $rp; kill $pp 2>/dev/null
    else
      pids=()
      for id in $(awk '/^\s*device:/{for(i=2;i<=NF;i++) print $i}' "$conf"); do
        sed "s/^\(\s*device:\).*/\1 $id/" "$conf" >"$d/$id.conf"
        "${pre[@]}" "$RVS_BIN" -c "$d/$id.conf" >"$d/rvs_$id.log" 2>&1 & pids+=($!)
      done
      wait "${pids[@]}"
    fi
    kill $sp 2>/dev/null; wait 2>/dev/null )
}

peak() {  # dir -> "ngpu sum mean min max"   (peak per-GPU sample, the sweep's metric)
  cat "$1"/rvs*.log 2>/dev/null | awk '/GFLOPS/{for(i=1;i<=NF;i++) if($i=="GFLOPS"){v=$(i+1)/1000}
      if(match($0,/GPU:+ *[0-9]+/)){g=substr($0,RSTART,RLENGTH); gsub(/[^0-9]/,"",g)} else next
      if(g!="0" && v>p[g]) p[g]=v}
    END{n=0;s=0;mn=1e12;mx=0; for(k in p){n++;s+=p[k]; if(p[k]<mn)mn=p[k]; if(p[k]>mx)mx=p[k]}
        if(n) printf "%d %.0f %.0f %.0f %.0f\n",n,s,s/n,mn,mx; else print "0 0 0 0 0"}'
}

# ---- stage 1 --------------------------------------------------------------------------
echo "stack arm rep ngpu sum mean min max" >"$OUT/stage1.txt"
for rep in $(seq 1 "$REPS"); do
  for stack in 7.14 host; do
    for a in n1 n2 n4 n8 m8; do
      case $a in n1) c=1x;; n2) c=2x;; n4) c=4x;; *) c=8x;; esac
      mode=single; [[ $a == m8 ]] && mode=multi
      d=$OUT/stage1/$stack/$a/r$rep
      run_rvs "$d" "$stack" "$SWEEP/${c}_fp4.conf" "$mode"
      r="$stack $a $rep $(peak "$d")"; echo "$r" >>"$OUT/stage1.txt"; say "stage1 $r"
    done
  done
done
verdict=$($PY - "$OUT/stage1.txt" <<'E'
import sys, collections
m = collections.defaultdict(list)
for l in open(sys.argv[1]).read().split("\n")[1:]:
    p = l.split()
    if len(p) == 8 and p[3] != "0": m[(p[0], p[1])].append(float(p[5]))
g = lambda k: sum(m[k]) / len(m[k]) if m[k] else float("nan")
n1, n8, m8 = g(("7.14", "n1")), g(("7.14", "n8")), g(("7.14", "m8"))
ok = n8 < 0.85 * n1 and m8 >= 0.90 * n1
print(f"{'CONFIRMED' if ok else 'NOT_CONFIRMED'} 7.14: n1={n1:.0f} n8={n8:.0f} ({n8/n1:.2f}x) m8={m8:.0f} ({m8/n1:.2f}x)")
E
)
say "VERDICT $verdict"

# ---- stage 2 --------------------------------------------------------------------------
if [[ $verdict == CONFIRMED* ]]; then
  for c in 1x 8x; do sed 's/^\(\s*duration:\).*/\1 10000/' "$SWEEP/${c}_fp4.conf" >"$OUT/${c}_fp4_10s.conf"; done
  prof() {  # name stack conf mode
    local d=$OUT/stage2/$1 rp
    rp=$( [[ $2 == 7.14 ]] && echo "$SW/rocm-7.14.0/bin/rocprofv3" || echo /opt/rocm/bin/rocprofv3 )
    run_rvs "$d" "$2" "$3" "$4" "$rp" --kernel-trace --output-format csv -d "$d/trace" -o "%pid%" --
    say "stage2 $1: $(peak "$d") traces=$(find "$d/trace" -name '*kernel_trace.csv' | wc -l)"
  }
  prof n1_714 7.14 "$OUT/1x_fp4_10s.conf" single
  prof n8_714 7.14 "$OUT/8x_fp4_10s.conf" single
  prof m8_714 7.14 "$OUT/8x_fp4_10s.conf" multi
  prof n8_724 host "$OUT/8x_fp4_10s.conf" single
  # host hot spots of the 7.14 single-process n8 run
  if [[ $(cat /proc/sys/kernel/perf_event_paranoid) -le 1 ]]; then
    d=$OUT/stage2/perf_n8_714; mkdir -p "$d"
    ( export ROCM_STACK=7.14; source $B/common/env.sh >/dev/null 2>&1
      "$RVS_BIN" -c "$SWEEP/8x_fp4.conf" >"$d/rvs.log" 2>&1 & rp=$!
      sleep 12; perf record -g -F 499 -p $rp -o "$d/perf.data" -- sleep 10 >"$d/perf_record.log" 2>&1
      wait $rp
      perf report -i "$d/perf.data" --no-children --sort comm,dso,sym --stdio 2>/dev/null | head -80 >"$d/perf_top.txt" )
    say "stage2 perf done"
  else
    say "stage2 perf skipped (perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid))"
  fi
else
  say "not confirmed: no profiling"
fi
flock -u 9; exec 9>&-

# ---- report ---------------------------------------------------------------------------
$PY $B/work-rocmval/analyze_fp4_recheck.py "$OUT" -o $B/results/vs-amd-cloud/rvs-fp4-recheck.md >"$OUT/analyze.log" 2>&1
say "analyze rc=$? -> results/vs-amd-cloud/rvs-fp4-recheck.md"
say "FP4 RECHECK DONE"
