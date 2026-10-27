#!/usr/bin/env bash
# ROCm 10.1.0 (+ hipBLASLt 1.4.1, RCCL 2.30.7) vs the current ROCm 7.14 results, node6100 (+ node6101
# for the 2-node tests). Plan: ../plan-new-version.txt. Only the tests the update affects:
#   1. hipBLASLt MX GEMM correctness (hipblaslt-bench -v), on 7.14 and on 10.1
#   2. RVS gst GEMM, 9 precisions, 1..8 GPUs          (same as work-rocmval/run_part_a.sh stages 1-2)
#   3. RCCL bandwidth, 1 node                          (rccl-tests/run_part_b.sh)
#   4. RCCL latency, 1 node                            (run_rccl_lat_1node_rocm10.1.sh)
#   5. rocHPL fp64, 1..8 GPUs                          (run_rochpl_rocm10.1.sh)
#   6. rocHPL fp64, 16 GPUs on 2 nodes
#   7. RCCL bandwidth, 2 nodes                         (rccl-tests/run_part_b_2node.sh)
#   8. RCCL latency, 2 nodes                           (run_rccl_lat_2node_rocm10.1.sh)
#   9. compare_rocm10.1.py -> results/rocm10.1-vs-rocm7.14.md (no existing md file is changed)
# A stage that fails is logged and skipped; the comparison marks it "not run".
# Waits for amd-software/setup/install-rocm-10.1.sh (rocm-10.1.0.READY) first.
# Run from node6100 (lock order 6100 -> 6101):
#   ./launch.sh node6100 rocm10.1/run_rocm10.1_all.sh
# Writes $STATE_DIR/DONE at the end; the login-node waiter (push_when_done.sh) then syncs and pushes.
set -uo pipefail
export ROCM_STACK=10.1
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
HERE=$BENCH_ROOT/rocm10.1
STATE_DIR=$LOG_ROOT/driver; mkdir -p "$STATE_DIR"
STATE=$STATE_DIR/STATE.txt
rm -f "$STATE_DIR/DONE"
say() { echo "[$(date -Iseconds)] $*" | tee -a "$STATE"; }
say "ROCm 10.1 driver start on $HOSTNAME_S, results=$RESULTS logs=$LOG_ROOT"
[[ $HOSTNAME_S == node6100 ]] || { say "ABORT: run from node6100"; exit 1; }
LOCK=/dev/shm/shaohao-gpu.lock
PEER=${PEER:-node6101.inband}
STAGES=${STAGES:-mx rvs rccl lat1 hpl hpl16 rccl2 lat2}
has() { [[ " $STAGES " == *" $1 "* ]]; }

# ---- 0. wait for the install -----------------------------------------------------------
for i in $(seq 1 480); do   # up to 8 h
  [[ -f $SW/rocm-10.1.0.READY ]] && break
  if (( i % 4 == 0 )) && ! pgrep -f install-rocm-10.1.sh >/dev/null; then
    [[ -f $SW/rocm-10.1.0.READY ]] && break
    say "ABORT: install stopped without READY. Last lines of $SW/setup/logs/install-rocm-10.1.log:"
    tail -20 "$SW/setup/logs/install-rocm-10.1.log" | tee -a "$STATE"
    echo "install failed" > "$STATE_DIR/DONE"; exit 1
  fi
  sleep 60
done
[[ -f $SW/rocm-10.1.0.READY ]] || { say "ABORT: install not done after 8 h"; echo "install timeout" > "$STATE_DIR/DONE"; exit 1; }
say "install READY ($(cat "$SW/rocm-10.1.0.READY"))"

# ---- versions (both stacks) ------------------------------------------------------------
ver() {  # $1 = rocm dir
  local r=$1
  echo "hip=$("$r/bin/hipconfig" --version 2>/dev/null)"
  echo "rccl=$(awk '/define NCCL_MAJOR/{a=$3}/define NCCL_MINOR/{b=$3}/define NCCL_PATCH/{c=$3}END{print a"."b"."c}' "$r/include/rccl/rccl.h")"
  echo "hipblaslt=$(cat "$r"/include/hipblaslt/*.h | awk '/define HIPBLASLT_VERSION_MAJOR/{a=$3}/define HIPBLASLT_VERSION_MINOR/{b=$3}/define HIPBLASLT_VERSION_PATCH/{c=$3}END{print a"."b"."c}')"
  echo "rocblas=$(cat "$r"/include/rocblas/internal/rocblas-version.h 2>/dev/null | awk '/define ROCBLAS_VERSION_MAJOR/{a=$3}/define ROCBLAS_VERSION_MINOR/{b=$3}/define ROCBLAS_VERSION_PATCH/{c=$3}END{print a"."b"."c}')"
  echo "clang=$("$r/llvm/bin/clang" --version 2>/dev/null | head -1)"
}
{ echo "[rocm7.14]"; echo "rocm=7.14.0"; ver "$SW/rocm-7.14.0"
  echo "[rocm10.1]"; echo "rocm=10.1.0"; ver "$SW/rocm-10.1.0"
  echo "[host]"; echo "driver=$(cat /sys/module/amdgpu/version 2>/dev/null)"; echo "kernel=$(uname -r)"
  echo "ucx=$("$SW/ompi-5.0.8-rocm10.1/ucx/bin/ucx_info" -v 2>/dev/null | awk '/Library version/{print $NF}')"
  echo "openmpi=$("$SW/ompi-5.0.8-rocm10.1/openmpi/bin/mpirun" --version 2>/dev/null | head -1)"
  echo "rochpl=$(git -C /orcd/data/orcd/022/benchmarks/rochpl/src/rocHPL log -1 --format=%h)"
  echo "rvs_commit=c82e4aeb"
} > "$RESULTS/versions.txt" 2>&1
cat "$RESULTS/versions.txt" >> "$STATE"

# ---- node6100 lock ----------------------------------------------------------------------
say "waiting for $LOCK on $HOSTNAME_S"
exec 9>>"$LOCK"; flock 9
export GPU_LOCKS_HELD=1
say "got $HOSTNAME_S lock"
assert_gpus_idle | tee -a "$STATE"
pause() { sleep 30; }   # let VRAM drain between stages

# ---- 1. MX GEMM correctness ---------------------------------------------------------------
if has mx; then
  say "1. hipBLASLt MX GEMM check (7.14 and 10.1)"
  bash "$HERE/run_mx_check.sh" > "$STATE_DIR/mx.log" 2>&1; say "1. rc=$? -> $RESULTS/mx_check.csv"; pause
fi

# ---- 2. RVS gst sweep ------------------------------------------------------------------------
if has rvs; then
  cd "$BENCH_ROOT/work-rocmval"
  TS=$(date +%Y%m%d_%H%M%S)
  SMOKE=$LOG_ROOT/rvs/smoke_$TS SWEEP=$LOG_ROOT/rvs/sweep_$TS
  say "2. RVS smoke: 1 GPU x fp16 x 10s (rvs $RVS_BIN)"
  GPU_COUNTS=1 PRECISIONS=fp16 DURATION_MS=10000 OUT_DIR="$SMOKE" ./run_tflops.sh > "$STATE_DIR/rvs_smoke.log" 2>&1
  agg=$(awk -F, 'NR==2{print $(NF-1)}' "$SMOKE/summary.csv" 2>/dev/null)
  if [[ -z $agg || $agg == 0.00 ]]; then
    say "2. RVS smoke produced no TFLOPS -- skip RVS. See $STATE_DIR/rvs_smoke.log"
  else
    say "2. smoke OK ($agg TFLOPS); sweep N=1..8 x 9 precisions x 30s"
    GPU_COUNTS="1 2 3 4 5 6 7 8" DURATION_MS=30000 OUT_DIR="$SWEEP" ./run_tflops.sh > "$STATE_DIR/rvs_sweep.log" 2>&1
    say "2. sweep rc=$? rows=$(( $(wc -l < "$SWEEP/summary.csv" 2>/dev/null || echo 1) - 1 ))"
    $PY analyze_rvs.py "$SWEEP" -o "$RESULTS" > "$STATE_DIR/rvs_analyze.log" 2>&1
    say "2. analyze rc=$? -> $RESULTS/rvs_tflops.{md,csv}"
    # analyze_rvs.py has the host stack's name in its header; name the stack that ran
    sed -i 's/ROCm 7\.2\.4/ROCm 10.1.0/g' "$RESULTS/rvs_tflops.md" 2>/dev/null
  fi
  cd "$HERE"; pause
fi

# ---- 3. RCCL bandwidth, 1 node -------------------------------------------------------------
if has rccl; then
  say "3. RCCL 1-node bandwidth (run_part_b.sh, $RCCL_TESTS_DIR)"
  bash "$BENCH_ROOT/rccl-tests/run_part_b.sh" > "$STATE_DIR/rccl_1node.log" 2>&1; say "3. rc=$? -> $RESULTS/rccl.csv"; pause
fi

# ---- 4. RCCL latency, 1 node ------------------------------------------------------------------
if has lat1; then
  say "4. RCCL 1-node latency"
  bash "$HERE/run_rccl_lat_1node_rocm10.1.sh" > "$STATE_DIR/lat_1node.log" 2>&1; say "4. rc=$? -> $RESULTS/rccl_lat_1node.md"; pause
fi

# ---- 5. rocHPL 1..8 GPUs ------------------------------------------------------------------------
HPLOUT=$LOG_ROOT/rochpl/rochpl_$(date +%Y%m%d_%H%M%S); mkdir -p "$HPLOUT"
echo "$HPLOUT" > "$RESULTS/rochpl_run_dir.txt"
hpl_point() {   # $1 = G ; retry at 0.6 like run_rochpl_all.sh
  OUT=$HPLOUT bash "$HERE/run_rochpl_rocm10.1.sh" "$1" 0.75 | tee -a "$STATE" && return 0
  say "G=$1 failed at mem_frac 0.75 -> retry at 0.6"
  OUT=$HPLOUT bash "$HERE/run_rochpl_rocm10.1.sh" "$1" 0.6 | tee -a "$STATE"
}
HPL_SMOKE_OK=0
if has hpl || has hpl16; then
  say "5. rocHPL smoke: G=1 N=46080"
  if OUT=$HPLOUT N=46080 IT=1 bash "$HERE/run_rochpl_rocm10.1.sh" 1 | tee -a "$STATE"; then
    HPL_SMOKE_OK=1; sed -i '$d' "$HPLOUT/summary.txt"
  else
    say "5. rocHPL smoke failed -- skip rocHPL. Last lines:"; tail -30 "$HPLOUT"/rochpl_g1_N46080_*.log | tee -a "$STATE"
    sed -i '$d' "$HPLOUT/summary.txt" 2>/dev/null
  fi
fi
if has hpl && (( HPL_SMOKE_OK )); then
  for G in 1 2 3 4 5 6 7 8; do say "5. rocHPL G=$G"; hpl_point "$G"; done
  pause
fi

# ---- both nodes ---------------------------------------------------------------------------------
if has hpl16 || has rccl2 || has lat2; then
  say "waiting for $LOCK on $PEER"
  coproc PEERLOCK { ssh -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 "$PEER" \
                    "flock $LOCK -c 'echo LOCKED; cat >/dev/null'"; }
  read -r got <&"${PEERLOCK[0]}"
  if [[ $got == LOCKED ]]; then
    say "got $PEER lock -- both nodes held"
    if has hpl16 && (( HPL_SMOKE_OK )); then
      say "6. rocHPL smoke: G=16 N=131072"
      if OUT=$HPLOUT N=131072 IT=1 bash "$HERE/run_rochpl_rocm10.1.sh" 16 | tee -a "$STATE"; then
        sed -i '$d' "$HPLOUT/summary.txt"; say "6. rocHPL G=16"; hpl_point 16
      else
        say "6. G=16 smoke failed. Last lines:"; tail -30 "$HPLOUT"/rochpl_g16_N131072_*.log | tee -a "$STATE"
        sed -i '$d' "$HPLOUT/summary.txt"
      fi
      pause
    fi
    if has rccl2; then
      say "7. RCCL 2-node bandwidth (run_part_b_2node.sh, $RCCL_TESTS_MPI_DIR)"
      bash "$BENCH_ROOT/rccl-tests/run_part_b_2node.sh" > "$STATE_DIR/rccl_2node.log" 2>&1; say "7. rc=$? -> $RESULTS/rccl_2node.csv"; pause
    fi
    if has lat2; then
      say "8. RCCL 2-node latency"
      bash "$HERE/run_rccl_lat_2node_rocm10.1.sh" > "$STATE_DIR/lat_2node.log" 2>&1; say "8. rc=$? -> $RESULTS/rccl_lat.md"
    fi
    exec {PEERLOCK[1]}>&-
  else
    say "could not take $LOCK on $PEER -- skip the 2-node tests"
  fi
fi
exec 9>&-
say "tests done, locks released"

# ---- 9. comparison ------------------------------------------------------------------------------
$PY "$HERE/compare_rocm10.1.py" > "$STATE_DIR/compare.log" 2>&1
say "9. compare rc=$? -> $BENCH_ROOT/results/rocm10.1-vs-rocm7.14.md"
date -Iseconds > "$STATE_DIR/DONE"
say "ROCm 10.1 driver DONE"
