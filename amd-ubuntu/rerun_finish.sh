#!/usr/bin/env bash
# 2026-10-05: finish the reruns the agents started. Per node, each GPU job holds /dev/shm/shaohao-gpu.lock.
#   ./launch.sh node6100 rerun_finish.sh   (node6100: megatron-ref, then waits for everything, then report.py)
#   ./launch.sh node6101 rerun_finish.sh   (node6101: Kimi recipe C=14 retry)
B=/orcd/data/orcd/022/benchmarks/amd-ubuntu; cd "$B"; source common/env.sh
say() { echo "[$(date -Iseconds)] $(hostname -s) $*"; }
busy() { for n in node6100 node6101; do ssh -o BatchMode=yes $n.inband \
  'pgrep -f "run_part_b_2node|run_megatron_ref|run_kimi_recipe|run_part_d|run_kimi_all" >/dev/null' && return 0; done; return 1; }
case $(hostname -s) in
node6100*)
  say "megatron-ref N=8 (waiting for GPU lock)"
  flock /dev/shm/shaohao-gpu.lock megatron-ref/run_megatron_ref.sh 8; say "megatron-ref rc=$?"
  until ! busy; do sleep 300; done
  say "all reruns finished; report.py"
  flock logs/auto/report.lock $PY report.py; say "report rc=$?" ;;
node6101*)
  say "Kimi recipe C=14 retry (waiting for GPU lock)"
  flock /dev/shm/shaohao-gpu.lock atom/run_kimi_recipe.sh 14; say "recipe C=14 rc=$?"
  flock logs/auto/report.lock $PY report.py ;;
esac
