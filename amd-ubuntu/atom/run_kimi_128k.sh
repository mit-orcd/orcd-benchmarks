#!/usr/bin/env bash
# Run one Kimi-K3 recipe on AMD's 128K/1K workload (amd-kimi-k3-recipe.pdf: ISL/OSL 128000/1000,
# range ratio 0.2, C = 1..128, 10*C prompts) so it can be compared with AMD's published numbers:
#   ./launch.sh node6100 atom/run_kimi_128k.sh atom       # ATOM (base recipe, max-num-seqs 64)
#   ./launch.sh node6101 atom/run_kimi_128k.sh recipe1    # vLLM recipe 1 (run_kimi_recipe.sh)
# Waits until no other user is on the GPUs, takes the node GPU lock, runs, releases the lock,
# then writes results/<node>/kimi-128k/<arm>.{csv,md} and regenerates the comparison md files.
set -uo pipefail
ARM=${1:?usage: $0 atom|recipe1}
export RESULTS_SUBDIR=kimi-128k
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/gpu_wait.sh
cd "$(dirname "$0")"
W=$LOG_ROOT/wrapper_${ARM}_$(date +%Y%m%d_%H%M%S).log
say() { echo "[$(date -Iseconds)] $*" | tee -a "$W"; }
say "Kimi-K3 128K/1K, arm=$ARM on $HOSTNAME_S"

wait_for_others
take_gpu_lock
wait_for_others            # re-check after our own earlier jobs released the lock

case $ARM in
  atom)
    bash run_kimi_128k_atom.sh >>"$W" 2>&1
    SWEEPS=$(ls -d "$LOG_ROOT"/atom/kimi_128k_atom_* 2>/dev/null) ;;
  recipe1)
    RESULTS_SUBDIR=kimi-recipe-128k ISL=128000 OSL=1000 RR=0.2 WARM_MAX=4 BENCH_TIMEOUT=36000 \
      JSON_PREFIX=Kimi-K3-MXFP4_isl128000_osl1000_ CONC="1 2 4 8 16 32 64 128" \
      bash run_kimi_recipe.sh >>"$W" 2>&1
    SWEEPS=$(ls -d "$BENCH_ROOT/logs/$HOSTNAME_S/kimi-recipe-128k/atom/kimi_recipe_"* 2>/dev/null) ;;
  *) say "unknown arm $ARM"; exit 1 ;;
esac
say "run finished rc=$?; releasing the GPU lock"
flock -u 9; exec 9>&-

$PY analyze_kimi_amdrecipe.py "$ARM" $SWEEPS -o "$RESULTS" --title "$ARM on AMD's 128K/1K workload" >>"$W" 2>&1
say "analyze rc=$? -> $RESULTS/$ARM.{csv,md}"
flock "$BENCH_ROOT/logs/auto/report.lock" $PY compare_kimi_amdrecipe.py >>"$W" 2>&1
flock "$BENCH_ROOT/logs/auto/report.lock" $PY "$BENCH_ROOT/report.py" >>"$W" 2>&1
say "comparison md files regenerated; DONE"
