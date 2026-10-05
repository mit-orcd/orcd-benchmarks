#!/usr/bin/env bash
# Run every single-node part on THIS node, in order, unattended:
#   A RVS -> B rccl-tests -> C Primus (incl. Megatron) -> megatron-ref -> D ATOM (tiers 1-3)
# Launch from a login node, one per node (both nodes can run at the same time):
#   ./launch.sh node6100 run_all_node.sh            # all parts
#   ./launch.sh node6101 run_all_node.sh "A B"      # just RVS + rccl-tests
# The 2-node RCCL run needs both nodes idle, so it is separate: rccl-tests/run_part_b_2node.sh.
#
# Each part is skipped (not failed) when its prerequisite is missing, so this can be started
# before the container images / apptainer / Kimi mount exist and re-run later for the rest.
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$BENCH_ROOT"
PARTS="${1:-A B C M D}"
mkdir -p "$LOG_ROOT"
STATE=$LOG_ROOT/run_all_$(date +%Y%m%d_%H%M%S).txt
say() { echo "[$(date -Iseconds)] $HOSTNAME_S $*" | tee -a "$STATE"; }
# apptainer may be installed but unusable (AppArmor blocks unprivileged user namespaces), so try it.
have_ctr() { [[ -s $1 ]] && apptainer exec "$1" true >/dev/null 2>&1; }

say "run_all_node start: parts='$PARTS' rocm=$ROCM_STACK ($ROCM_PATH) results=$RESULTS"
if ! assert_gpu_access; then say "ABORT: no GPU access (render/video group)"; exit 1; fi

for p in $PARTS; do
  case $p in
    A) say "PART A (RVS)";        work-rocmval/run_part_a.sh; say "PART A rc=$?" ;;
    B) [[ -x $RCCL_TESTS_DIR/all_reduce_perf ]] || { say "SKIP B: rccl-tests not built ($RCCL_TESTS_DIR)"; continue; }
       say "PART B (rccl-tests)"; rccl-tests/run_part_b.sh; say "PART B rc=$?" ;;
    C) have_ctr "$PRIMUS_SIF" || { say "SKIP C: needs apptainer + $PRIMUS_SIF"; continue; }
       say "PART C (Primus)";     primus/run_part_c.sh; say "PART C rc=$?" ;;
    M) have_ctr "$MEGATRON_SIF" || { say "SKIP megatron-ref: needs apptainer + $MEGATRON_SIF"; continue; }
       say "megatron-ref (GPT-15.6B, 8 GPU)"; megatron-ref/run_megatron_ref.sh 8; say "megatron-ref rc=$?" ;;
    D) have_ctr "$SIF_DIR/atom-dev-${ATOM_IMG_CLOUD#*:}.sif" || { say "SKIP D: needs apptainer + the $ATOM_IMG_CLOUD SIF"; continue; }
       # Tiers 1-2 on amd-cloud's exact image (apple-to-apple). Tier 3 (Kimi-K3) runs in
       # atom/run_kimi_all.sh, once per image set, after the /scratch/Kimi-K3 copy is complete.
       say "PART D (ATOM tiers 1-2, $ATOM_IMG_CLOUD)"
       ATOM_IMG=$ATOM_IMG_CLOUD atom/run_part_d.sh "tier1 tier2"; say "PART D rc=$?" ;;
    *) say "unknown part '$p' (use A B C M D)" ;;
  esac
  # Refresh results/ubuntu/ and results/vs-amd-cloud/ as soon as each benchmark finishes.
  $PY "$BENCH_ROOT/report.py" >> "$STATE" 2>&1 || say "report.py failed after $p"
  sleep 30   # let VRAM drain before the next part
done
say "run_all_node DONE"
