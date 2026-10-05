#!/usr/bin/env bash
# Rerun Part C stage 3 (Primus Megatron llama2-7B, N=1..8) + stage 4 report, reusing the
# original Part C RUN_ID so the microbench results stay in the same report.
#
# 2026-10-05 fix: the first run failed at every N with ENOSPC: Primus JIT-builds apex's
# fused_weight_gradient_mlp_cuda inside /opt/venv, which under the shim's default
# --writable-tmpfs is capped at 64 MB. Use a persistent per-node overlay instead (one
# container at a time per overlay; per-node name so the two nodes never share it).
# Run under the node GPU lock: flock /dev/shm/shaohao-gpu.lock primus/rerun_megatron_llama.sh
set -uo pipefail
# take the node GPU lock (one GPU job per node at a time)
[[ -n ${_GPU_LOCKED:-} ]] || exec env _GPU_LOCKED=1 flock /dev/shm/shaohao-gpu.lock bash "$0" "$@"
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$BENCH_ROOT/primus"
say() { echo "[$(date -Iseconds)] $(hostname -s) $*"; }

DRV=$(ls -dt "$LOG_ROOT"/primus/part_c_* | tail -n 1)          # the original Part C run
RUN_ID=$(sed -n 's/^RUN_ID=//p' "$DRV/STATE.txt" | tail -n 1)
[[ -n $RUN_ID ]] || { say "no RUN_ID in $DRV/STATE.txt"; exit 1; }
export CTR_OVERLAY="primus-v26.5-$(hostname -s)"
say "Primus llama2-7B rerun RUN_ID=$RUN_ID overlay=$CTR_OVERLAY"

./run_megatron.sh "$RUN_ID" >"$DRV/megatron_rerun_$(date +%Y%m%d_%H%M%S).log" 2>&1
say "megatron rc=$?"
grep -E "OK|FAIL|TIMEOUT" "$LOG_ROOT/primus/sweep-$RUN_ID/summary.txt" | tail -n 8

$PY generate_report.py "$LOG_ROOT/primus/sweep-$RUN_ID" "$BENCH_ROOT/primus/sweep_out_$RUN_ID" \
    "$REF_ROOT/../dell-cloud/megatron-lm/summary.md" "$RESULTS/PRIMUS_REPORT.md" \
    >"$DRV/report_rerun.log" 2>&1
say "generate_report rc=$?"

# generate_report rewrites PRIMUS_REPORT.md; put the megatron-ref §1.2a block back.
MS=$(grep -l "update rc=0" "$LOG_ROOT"/megatron-ref/run_*/STATE.txt 2>/dev/null | sort | tail -n 1)
if [[ -n $MS ]]; then
  $PY "$BENCH_ROOT/megatron-ref/update_b200_table.py" "$MS" "$RESULTS/PRIMUS_REPORT.md" \
      >"$DRV/update_b200_rerun.log" 2>&1
  say "megatron-ref block rc=$? ($MS)"
fi
cd "$BENCH_ROOT" && flock logs/auto/report.lock $PY report.py && say "report.py done"
