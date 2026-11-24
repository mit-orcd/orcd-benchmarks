#!/usr/bin/env bash
# Nov 18: own-script check of the Hyperloom step 3 winner (session 20261117T201037Z-1a94cc7f,
# +13.08% at 64 users, no accuracy / TPOT / TTFT recorded by Hyperloom).
#   arm A: FP16 baseline   ATOM_GDN_SSM_DTYPE=fp16, batched 16384
#   arm B: step 3 setting  + ATOM_PREFILL_DECODE_INTERVAL=4 ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000, batched 8192
# Both: image nightly_202610011450, TP8, 64 users, ISL/OSL 1024/1024, 640 prompts, then GSM8K (all 1319).
# No Claude tokens, no push.   ./launch.sh node6101 hyperloom/run_check_s3.sh
set -uo pipefail
H=/orcd/data/orcd/022/benchmarks/amd-ubuntu
LOG=$H/logs/hyperloom/check-s3-$(hostname -s).state
say() { echo "[$(date -Iseconds)] $*" >>"$LOG"; }
say "check start"
run() {  # $1 tag, $2 envs, $3 batched
  say "$1: start (envs '$2', batched $3)"
  ARMS="0:64" GSM8K=1 NO_REPORT=1 OUTTAG="check-s3-$1" EXTRA_ENVS="$2" BATCHED="$3" \
    bash "$H/atom/run_kimi_atom_mtp.sh" >>"$LOG.out" 2>&1
  local rc=$?
  local d; d=$(ls -1dt "$H"/logs/*/kimi-atom-mtp/atom/kimi_atom_mtp_check-s3-"$1"_* 2>/dev/null | head -1)
  say "$1: rc=$rc out=$d"
  [[ -n $d ]] && grep -E 'c=64 rc|GSM8K rc|FAILED|ABORT|died' "$d/STATE.txt" | sed "s/^/  $1: /" >>"$LOG"
  [[ -n $d && -f $d/k0/c64.json && ! -f $d/k0/FAILED ]]
}
for tag in A B; do
  if [[ $tag == A ]]; then e="ATOM_GDN_SSM_DTYPE=fp16"; b=16384
  else e="ATOM_GDN_SSM_DTYPE=fp16 ATOM_PREFILL_DECODE_INTERVAL=4 ATOM_PREFILL_DELAYER_MAX_QUEUE_MS=5000"; b=8192; fi
  run "$tag" "$e" "$b" || { say "$tag failed, retry once"; run "$tag" "$e" "$b" || { say "$tag failed twice -> stop"; exit 1; }; }
done
say "check DONE"
