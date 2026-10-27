#!/usr/bin/env bash
# ROCm 10.1 copy of ../work-rocmval/fp4_multiproc_test.sh: FP4 at N = 8 with one RVS process per GPU
# (8 processes started together) and the standard single process, both ROCm 10.1, same conf as the sweep.
#   ./launch.sh node6100 rocm10.1/fp4_multiproc_rocm10.1.sh
set -uo pipefail
[[ -n ${_GPU_LOCKED:-} ]] || exec env _GPU_LOCKED=1 flock /dev/shm/shaohao-gpu.lock bash "$0" "$@"
B=/orcd/data/orcd/022/benchmarks/amd-ubuntu
H=$(hostname -s); OUT=$B/logs/$H/rocm10.1/rvs/fp4_multiproc_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
CONF=$(ls -d $B/logs/$H/rocm10.1/rvs/sweep_*/8x_fp4.conf | head -n 1)
IDS=$(awk '/device:/{for(i=2;i<=NF;i++) print $i}' "$CONF")
say() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/summary.txt"; }
arm() {  # name mode
  local name=$1 mode=$2 pids=()
  ( export ROCM_STACK=10.1; source $B/common/env.sh >/dev/null 2>&1
    mkdir -p "$OUT/$name"; echo "RVS_BIN=$RVS_BIN" >"$OUT/$name/rvs_bin.txt"
    if [[ $mode == single ]]; then
      "$RVS_BIN" -c "$CONF" >"$OUT/$name/all.log" 2>&1
    else
      for id in $IDS; do
        sed "s/^\(\s*device:\).*/\1 $id/" "$CONF" >"$OUT/$name/$id.conf"
        "$RVS_BIN" -c "$OUT/$name/$id.conf" >"$OUT/$name/$id.log" 2>&1 & pids+=($!)
      done
      wait "${pids[@]}"
    fi )
  cat "$OUT/$name"/*.log | awk -v n="$name" '/GFLOPS/ && $5 ~ /^[0-9]/{g=$5; gsub(/]/,"",g); v=$7/1000; if(v>p[g]) p[g]=v}
    END{s=0; c=0; line=""; for(k in p){s+=p[k]; c++; line=line sprintf(" %s:%d",k,p[k])}
        printf "%-10s GPUs=%d sum=%.0f mean/GPU=%.0f |%s\n", n, c, s, (c?s/c:0), line}' | tee -a "$OUT/summary.txt"
}
say "fp4 multi-process test on $H, ROCm 10.1, conf $CONF"
arm multi101  multi
arm single101 single
say "done -> $OUT"
