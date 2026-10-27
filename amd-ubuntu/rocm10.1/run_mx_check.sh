#!/usr/bin/env bash
# hipBLASLt MX (microscaling) GEMM check on ROCm 7.14 and ROCm 10.1, 1 GPU (GPU 0).
# hipblaslt-bench is AMD's own, from each release's gfx950 test tarball
# (amd-software/hipblaslt-bench-<ver>, setup/install-rocm-10.1.sh); the libraries come from
# amd-software/rocm-<ver>/lib, so each run uses that release's hipBLASLt.
# Per case: (a) correctness, 2048^3 with -v (checked against the CPU reference): PASS when
# norm_error <= the bench's own rtol; (b) speed, 8192^3, 50 iterations, no check.
# Scale mode 3 = B32E8: one E8M0 scale per 32 elements (OCP MX). trig_float init (MX allows
# hpl, trig_float or uniform_01; hpl gives ~50 % zeros for FP4).
# The caller holds the GPU lock (run_rocm10.1_all.sh). Writes $RESULTS/mx_check.csv.
set -uo pipefail
export ROCM_STACK=10.1
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
OUT=$LOG_ROOT/mx/mx_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
CSV=$RESULTS/mx_check.csv
echo "version,case,a_type,b_type,m,n,k,status,gflops,error,rtol,log" > "$CSV"
export HIP_VISIBLE_DEVICES=0
# case|A type|B type
CASES="MXFP8 (E4M3)|f8_r|f8_r
MXBF8 (E5M2)|bf8_r|bf8_r
MXFP6 (E2M3)|f6_r|f6_r
MXBF6 (E3M2)|bf6_r|bf6_r
MXFP4 (E2M1)|f4_r|f4_r
FP8 x FP4|f8_r|f4_r
FP4 x FP8|f4_r|f8_r"

col() {  # $1 log  $2 column name -> value from the bench's CSV result line
  awk -F, -v c="$2" '/^\[0\]:/ {sub(/^\[0\]:/,""); for (i=1;i<=NF;i++) if ($i==c) k=i; getline; gsub(/ /,""); print $k; exit}' "$1"
}

for VER in 7.14 10.1; do
  R=$SW/rocm-$VER.0 D=$SW/hipblaslt-bench-$VER.0
  if [[ ! -x $D/bin/hipblaslt-bench ]]; then
    echo "no hipblaslt-bench for $VER"; continue
  fi
  export LD_LIBRARY_PATH=$R/lib:$R/lib/host-math/lib:$R/lib/rocm_sysdeps/lib:$R/llvm/lib
  while IFS='|' read -r name a b; do
    tag=$(echo "$name" | tr -c 'A-Za-z0-9' '_' | tr -s _)
    common=(--transA T --transB N --a_type "$a" --b_type "$b" --c_type bf16_r --d_type bf16_r
            --compute_type f32_r --scaleA 3 --scaleB 3 --initialization trig_float)
    lv=$OUT/${VER}_${tag}_verify.log lp=$OUT/${VER}_${tag}_perf.log
    timeout 900 "$D/bin/hipblaslt-bench" -m 2048 -n 2048 -k 2048 "${common[@]}" -v -i 5 -j 1 > "$lv" 2>&1; rcv=$?
    timeout 900 "$D/bin/hipblaslt-bench" -m 8192 -n 8192 -k 8192 "${common[@]}" -i 50 -j 5 > "$lp" 2>&1; rcp=$?
    err=$(col "$lv" norm_error); rtol=$(col "$lv" rtol); gf=$(col "$lp" hipblaslt-Gflops)
    if grep -q 'Is supported 0\|No solution\|not supported' "$lv"; then st=UNSUPPORTED
    elif [[ $rcv -ne 0 || -z $err ]]; then st=ERROR
    elif awk -v e="$err" -v t="$rtol" 'BEGIN{exit !(e+0 <= t+0)}'; then st=PASS
    else st=FAIL; fi
    [[ $rcp -ne 0 ]] && gf=""
    echo "$VER,$name,$a,$b,2048/8192,2048/8192,2048/8192,$st,$gf,$err,$rtol,$lv" >> "$CSV"
    echo "$VER $name: $st err=$err rtol=$rtol gflops(8192)=$gf rc=$rcv/$rcp"
  done <<< "$CASES"
done
