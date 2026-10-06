#!/usr/bin/env bash
# fp4 N=8 test (2026-10-05): is the ROCm 7.14 drop at 8 GPUs (≈2,130 vs ≈4,080 TFLOPS/GPU at N=1)
# caused by driving all 8 GPUs from ONE rvs process (threads) rather than by the hardware?
#   arm multi714 : 8 separate rvs processes, one GPU each, ROCm 7.14, started together
#   arm single714: the original single-process 8-GPU conf, ROCm 7.14 (same-day control)
#   arm multi724 : 8 separate processes on host ROCm 7.2.4 (reference)
# Same conf as the sweep (8x_fp4.conf), only "device:" changed. Run on a node:
#   ./launch.sh node6100 work-rocmval/fp4_multiproc_test.sh
set -uo pipefail
[[ -n ${_GPU_LOCKED:-} ]] || exec env _GPU_LOCKED=1 flock /dev/shm/shaohao-gpu.lock bash "$0" "$@"
B=/orcd/data/orcd/022/benchmarks/amd-ubuntu
H=$(hostname -s); OUT=$B/logs/$H/rvs/fp4_multiproc_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
CONF=$(ls -d $B/logs/$H/rvs/sweep_*/8x_fp4.conf | head -n 1)
IDS=$(awk '/device:/{for(i=2;i<=NF;i++) print $i}' "$CONF")
say() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/summary.txt"; }

sample() {  # power/clock every 5 s while an arm runs
  while :; do echo "## $(date +%T)"; rocm-smi --showpower --showclocks 2>/dev/null | grep -E "Power|sclk"; sleep 5; done
}

arm() {  # name stack mode
  local name=$1 stack=$2 mode=$3 pids=() sp
  ( export ROCM_STACK=$stack; source $B/common/env.sh >/dev/null 2>&1
    mkdir -p "$OUT/$name"; echo "RVS_BIN=$RVS_BIN" >"$OUT/$name/rvs_bin.txt"
    sample >"$OUT/$name/smi.txt" 2>&1 & sp=$!
    if [[ $mode == single ]]; then
      "$RVS_BIN" -c "$CONF" >"$OUT/$name/all.log" 2>&1
    else
      for id in $IDS; do
        sed "s/^\(\s*device:\).*/\1 $id/" "$CONF" >"$OUT/$name/$id.conf"
        "$RVS_BIN" -c "$OUT/$name/$id.conf" >"$OUT/$name/$id.log" 2>&1 & pids+=($!)
      done
      wait "${pids[@]}"
    fi
    kill $sp 2>/dev/null )
  # peak per GPU (same metric as the sweep), then the sum
  cat "$OUT/$name"/*.log | awk -v n="$name" '/GFLOPS/ && $5 ~ /^[0-9]/{g=$5; gsub(/]/,"",g); v=$7/1000; if(v>p[g]) p[g]=v}
    END{s=0; c=0; line=""; for(k in p){s+=p[k]; c++; line=line sprintf(" %s:%d",k,p[k])}
        printf "%-10s GPUs=%d sum=%.0f mean/GPU=%.0f |%s\n", n, c, s, (c?s/c:0), line}' | tee -a "$OUT/summary.txt"
}

say "fp4 multi-process test on $H, conf $CONF"
arm multi714  7.14 multi
arm single714 7.14 single
arm multi724  host multi
say "done -> $OUT"
