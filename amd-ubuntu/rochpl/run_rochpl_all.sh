#!/usr/bin/env bash
# rocHPL driver: wait for the build -> smoke -> 1..8 GPUs on this node -> 16 GPUs on 2 nodes
# -> analysis (results/rochpl-vs-rvs.md) -> sync and push. Run from node6100:
#   ./launch.sh node6100 rochpl/run_rochpl_all.sh
# Knobs (env): GPUS ("1 2 3 4 5 6 7 8"), TWO_NODE (1), PUSH (1), plus run_rochpl.sh's.
# Each point runs at mem_frac 0.75; if it fails (e.g. out of memory) it is retried at 0.6.
set -uo pipefail
export ROCM_STACK=7.14   # amd-software/rocm-7.14.0 -> logs/<node>/rocm7.14/rochpl
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$(dirname "$0")"
R=/orcd/data/orcd/022/benchmarks/rochpl
GPUS=${GPUS:-1 2 3 4 5 6 7 8}
TWO_NODE=${TWO_NODE:-1}; PUSH=${PUSH:-1}
LOCK=/dev/shm/shaohao-gpu.lock
PEER=${PEER:-node6101.inband}
TS=$(date +%Y%m%d_%H%M%S)
export OUT=$LOG_ROOT/rochpl/rochpl_$TS; mkdir -p "$OUT"
STATE=$OUT/STATE.txt
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "rocHPL start on $HOSTNAME_S, out=$OUT"
[[ $HOSTNAME_S == node6100 ]] || { say "ABORT: run from node6100 (lock order 6100 -> 6101)"; exit 1; }

# ---- wait for the build (setup/build-rochpl.sh writes $R/BUILD_OK) ----------------------
for i in $(seq 1 240); do
  [[ -f $R/BUILD_OK ]] && break
  if (( i % 6 == 0 )) && ! pgrep -f build-rochpl.sh >/dev/null; then
    [[ -f $R/BUILD_OK ]] && break
    say "ABORT: build stopped without BUILD_OK. Last lines of $R/setup/logs/build.log:"
    tail -15 "$R/setup/logs/build.log" | tee -a "$STATE"; exit 1
  fi
  sleep 30
done
[[ -f $R/BUILD_OK ]] || { say "ABORT: build not done after 2 h"; exit 1; }
say "build OK ($(cat "$R/BUILD_OK"))"
# our Open MPI/UCX before amd-software/openmpi/lib (no UCX/ROCm) from the node env; ROCm lib for libamdhip64
export PATH=$ROCM_PATH/bin:$PATH LD_LIBRARY_PATH=$R/openmpi/lib:$R/ucx/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}
{ "$R/openmpi/bin/mpirun" --version | head -1; "$R/ucx/bin/ucx_info" -v | head -2
  git -C "$R/src/rocHPL" log -1 --format='rocHPL %h %ad'; echo "ROCm $ROCM_PATH $(hipconfig --version)"; echo
  "$R/ucx/bin/ucx_info" -d | grep -E 'Transport|Device' | paste - - | sort -u; } >"$OUT/versions.txt" 2>&1

run_point() {   # $1 = G ; retry at a smaller N if the first try fails
  ./run_rochpl.sh "$1" 0.75 | tee -a "$STATE" && return 0
  say "G=$1 failed at mem_frac 0.75 -> retry at 0.6"
  ./run_rochpl.sh "$1" 0.6 | tee -a "$STATE"
}

# ---- node6100 lock ---------------------------------------------------------------------
say "waiting for $LOCK on $HOSTNAME_S"
exec 9>>"$LOCK"; flock 9
say "got $HOSTNAME_S lock"
assert_gpus_idle | tee -a "$STATE"

# ---- smoke: 1 GPU, small N -------------------------------------------------------------
say "smoke: G=1 N=46080"
if ! N=46080 IT=1 ./run_rochpl.sh 1 | tee -a "$STATE"; then
  say "ABORT: smoke failed. Last lines:"; tail -30 "$OUT"/rochpl_g1_N46080_*.log | tee -a "$STATE"; exit 1
fi
sed -i '$d' "$OUT/summary.txt"   # smoke row is not a result

# ---- 1..8 GPUs on this node -------------------------------------------------------------
for G in $GPUS; do say "G=$G"; run_point "$G"; done

# ---- 16 GPUs on 2 nodes -----------------------------------------------------------------
if [[ $TWO_NODE == 1 ]]; then
  say "waiting for $LOCK on $PEER"
  coproc PEERLOCK { ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "$PEER" \
                    "flock $LOCK -c 'echo LOCKED; cat >/dev/null'"; }
  read -r got <&"${PEERLOCK[0]}"
  if [[ $got == LOCKED ]]; then
    say "got $PEER lock -- both nodes held"
    b=$(ssh -o BatchMode=yes "$PEER" "rocm-smi --showuse 2>/dev/null | awk '/GPU use/ {print \$NF}' | grep -cv '^0\$'")
    [[ "${b:-0}" -eq 0 ]] || say "WARNING: $b GPU(s) busy on $PEER"
    say "smoke: G=16 N=131072"
    if N=131072 IT=1 ./run_rochpl.sh 16 | tee -a "$STATE"; then
      sed -i '$d' "$OUT/summary.txt"
      say "G=16"; run_point 16
    else
      say "G=16 smoke failed (kept in summary.txt). Last lines:"
      tail -30 "$OUT"/rochpl_g16_N131072_*.log | tee -a "$STATE"
    fi
    exec {PEERLOCK[1]}>&-   # releases the peer's lock
  else
    say "could not take $LOCK on $PEER -- skip 16 GPUs"
  fi
fi
exec 9>&-
say "sweep done"

# ---- analysis + push ----------------------------------------------------------------------
mkdir -p "$BENCH_ROOT/logs/auto"
flock "$BENCH_ROOT/logs/auto/report.lock" $PY analyze_rochpl.py "$OUT" 2>&1 | tee -a "$STATE"
if [[ $PUSH == 1 ]]; then
  say "sync and push"
  /orcd/data/orcd/022/benchmarks/sync-and-push.sh "rocHPL on MI355X: 1-8 GPUs and 2 nodes, compared with RVS fp64

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >>"$OUT/push.log" 2>&1
  say "push rc=$? (log $OUT/push.log)"
fi
say "rocHPL DONE"
