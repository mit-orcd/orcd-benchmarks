#!/usr/bin/env bash
# Unattended watcher. Each node runs ONE job at a time (so runs never share GPUs); the two nodes
# run their single-node jobs in parallel. Per-node pipeline:
#   host-<n>  : GPU access on <n>                    -> run_all_node.sh "A B"     (RVS, RCCL, host ROCm 7.2.4)
#   2node     : host done on BOTH, both idle         -> rccl-tests/run_part_b_2node.sh (holds both)
#   ctr-<n>   : 2node done, apptainer + SIFs         -> run_all_node.sh "C M D"   (Primus, Megatron, ATOM)
#   kimi-<n>  : ctr-<n> done, /scratch/Kimi-K3 copy complete on <n>:
#                 node6100 -> atom/run_kimi_all.sh cloud   (amd-cloud's exact images)
#                 node6101 -> atom/run_kimi_recipe.sh      (AMD's latest vLLM recipe)
# End of the queue, the host benchmarks again on ROCm 7.14 (amd-cloud's ROCm, amd-software/rocm-7.14.0):
#   host714-<n>: kimi-<n> done                       -> with_rocm.sh 7.14 run_all_node.sh "A B"
#   2node714  : host714 done on BOTH, both idle      -> with_rocm.sh 7.14 rccl-tests/run_part_b_2node.sh
# Analysis: each part/experiment runs its own analyzer and report.py when it finishes; this
# watcher runs report.py again after every job (results/ubuntu/, results/vs-amd-cloud/).
# State: logs/auto/<job>.done when finished; logs/auto/running-<node> = "<job> <pid> <host>" while
# a job runs, so a restarted watcher re-attaches instead of launching a duplicate.
# Polls every 5 min for up to 14 days. Stop: touch logs/auto/STOP. Restart: ./launch.sh node6100 auto_run.sh
set -uo pipefail
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
A=$BENCH_ROOT/logs/auto; mkdir -p "$A"
LOG=$A/auto_run.log
say() { echo "[$(date -Iseconds)] $*" >> "$LOG"; }
NODES=(node6100 node6101)
declare -A KIMI_CMD=([node6100]="atom/run_kimi_all.sh cloud" [node6101]="atom/run_kimi_recipe.sh")
on() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$1.inband" "${@:2}"; }
done_() { [[ -f $A/$1.done ]]; }
report() { on node6100 "$PY $BENCH_ROOT/report.py" >> "$LOG" 2>&1 || say "report.py failed"; }
sif() { echo "$SIF_DIR/atom-dev-${1#rocm/atom-dev:}.sif"; }

# GPU access needs the render/video groups; if missing, add them with sudo (approved by the
# user) via amd-software/setup/grant-gpu-access.sh. A new ssh session then has access.
gpu_ok() {
  on "$1" 'test -r /dev/kfd -a -w /dev/kfd' && return 0
  on "$1" "$SW/setup/grant-gpu-access.sh" >> "$LOG" 2>&1
  on "$1" 'test -r /dev/kfd -a -w /dev/kfd'
}
ctr_ok() { [[ -s $PRIMUS_SIF && -s $MEGATRON_SIF && -s $(sif "$ATOM_IMG_CLOUD") ]] &&
           on "$1" "apptainer exec $PRIMUS_SIF true >/dev/null 2>&1"; }
rocm714_ok() { [[ -x $SW/rvs-7.14.0/bin/rvs ]] && ls "$RCCL_TESTS"/build-rocm7.14/all_reduce_perf "$RCCL_TESTS"/build-mpi-rocm7.14/all_reduce_perf >/dev/null 2>&1; }
both_free() { [[ ! -f $A/running-node6100 && ! -f $A/running-node6101 ]]; }
kimi_sifs_ok() {
  if [[ $1 == node6101 ]]; then [[ -s $VLLM_SIF && -d $DSPARK_MODEL ]]
  else [[ -s $(sif "$ATOM_IMG_CLOUD") && -s $(sif "$ATOM_IMG_CLOUD2") && -s $(sif "$MAD_IMG_CLOUD") ]]; fi
}
# Kimi-K3 copy complete: every shard in the index exists and total size unchanged since last poll.
declare -A KSIZE=()
kimi_ok() {
  local n=$1 sz
  sz=$(on "$n" "cd $KIMI_MODEL 2>/dev/null && [ -s model.safetensors.index.json ] &&
        $PY -c \"import json,os,sys; m=json.load(open('model.safetensors.index.json'))['weight_map']; sys.exit(any(not os.path.exists(f) for f in set(m.values())))\" &&
        find . -maxdepth 2 -type f -printf '%s\n' | awk '{s+=\$1} END{print s}'") || { KSIZE[$n]=""; return 1; }
  [[ -n $sz && $sz == "${KSIZE[$n]:-}" ]] && return 0
  KSIZE[$n]=$sz; return 1
}

# ---- running-job bookkeeping (persisted)
running() { [[ -f $A/running-$1 ]] && cat "$A/running-$1"; }          # -> "job pid host"
launch() {   # $1 job name, $2 host node, $3.. script+args; records it on every node in $NODES_HELD
  local job=$1 host=$2 out pid n; shift 2
  out=$("$BENCH_ROOT/launch.sh" "$host" "$@" 2>&1); pid=$(awk '/^pid/{print $2}' <<<"$out")
  if [[ -z $pid ]]; then say "FAILED to launch $job on $host: $out"; return 1; fi
  for n in ${NODES_HELD:-$host}; do echo "$job $pid $host" > "$A/running-$n"; done
  say "START $job on $host (pid $pid) $(grep '^log:' <<<"$out")"
}
# Clear finished jobs; returns 0 if node $1 is free.
free() {
  local r job pid host n
  r=$(running "$1") || return 0
  read -r job pid host <<<"$r"
  on "$host" "kill -0 $pid 2>/dev/null" && return 1
  for n in "${NODES[@]}"; do [[ $(running "$n") == "$r" ]] && rm -f "$A/running-$n"; done
  date -Iseconds > "$A/$job.done"; say "DONE  $job (pid $pid on $host)"
  report
  return 0
}

# One line per poll, logged only when it changes.
LAST=""
status() {
  local s="" n r
  for n in "${NODES[@]}"; do
    r=$(running "$n") && s+=" $n:running(${r%% *})" || {
      gpu_ok "$n" || s+=" $n:waiting(gpu/render-group)"; }
  done
  [[ $s != "$LAST" ]] && say "status:${s:- idle}"
  LAST=$s
}

say "auto_run start on $HOSTNAME_S (pid $$)"
end=$(( $(date +%s) + 14*86400 ))
while (( $(date +%s) < end )); do
  [[ -f $A/STOP ]] && { say "STOP file seen, exiting (running jobs continue)"; exit 0; }
  for n in "${NODES[@]}"; do free "$n" >/dev/null; done
  for n in "${NODES[@]}"; do
    [[ -f $A/running-$n ]] && continue
    r714=0; rocm714_ok && r714=1
    if ! done_ "host-$n"; then
      gpu_ok "$n" && launch "host-$n" "$n" run_all_node.sh "A B"
    elif ! done_ 2node; then
      if done_ host-node6100 && done_ host-node6101 && both_free; then
        NODES_HELD="node6100 node6101" launch 2node node6100 rccl-tests/run_part_b_2node.sh
      fi
    elif ! done_ "ctr-$n"; then
      ctr_ok "$n" && launch "ctr-$n" "$n" run_all_node.sh "C M D"
    elif ! done_ "kimi-$n"; then
      if kimi_sifs_ok "$n" && kimi_ok "$n"; then launch "kimi-$n" "$n" ${KIMI_CMD[$n]}; fi
    # ---- end of the queue: the same host benchmarks again on ROCm 7.14
    elif ! done_ "host714-$n"; then
      (( r714 )) && launch "host714-$n" "$n" with_rocm.sh 7.14 run_all_node.sh "A B"
    elif ! done_ 2node714; then
      if (( r714 )) && done_ host714-node6100 && done_ host714-node6101 && both_free; then
        NODES_HELD="node6100 node6101" launch 2node714 node6100 with_rocm.sh 7.14 rccl-tests/run_part_b_2node.sh
      fi
    fi
  done
  status
  if done_ kimi-node6100 && done_ kimi-node6101 && done_ host714-node6100 && done_ host714-node6101 && done_ 2node714; then
    say "all jobs done"; report; exit 0; fi
  sleep 300
done
say "auto_run timed out after 14 days"
