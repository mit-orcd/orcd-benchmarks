#!/usr/bin/env bash
# plan-hyperloom.md step 3: our own same-day baselines, at 1/8/64/128/256 users, twice each.
#   A = ATOM no MTP (K=0), C = ATOM + MTP (K=6 at 1 user, K=2 above), B = vLLM AMD recipe (isl1k).
# Separate output (RESULTS_SUBDIR=hyperloom-baseline); published tables are not touched, no push.
#   ./launch.sh node6101 hyperloom/run_baselines.sh
set -uo pipefail
H=/orcd/data/orcd/022/benchmarks/amd-ubuntu
export RESULTS_SUBDIR=hyperloom-baseline NO_REPORT=1
S=$H/logs/hyperloom/baselines-$(hostname -s).state; mkdir -p "$(dirname "$S")"
say() { echo "[$(date -Iseconds)] $*" | tee -a "$S"; }
say "baselines start"
for rep in 1 2; do
  say "ATOM rep $rep"
  GREEDY_CHECK=$H/hyperloom/greedy_capture.py ARMS="0:1 8 64 128 256;6:1;2:8 64 128 256" \
    bash "$H/atom/run_kimi_atom_mtp.sh"; say "ATOM rep $rep rc=$?"
done
for rep in 1 2; do
  say "vLLM rep $rep"
  CONC="1 8 64 128 256" bash "$H/atom/run_kimi_amdrecipe.sh" isl1k; say "vLLM rep $rep rc=$?"
done
say "baselines DONE"
