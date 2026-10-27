#!/usr/bin/env bash
# RVS gst on ROCm 10.1 on this node, N = 1 and 8, all 9 precisions (same settings as the sweep),
# so MI355X numbers can be the mean of node6100 and node6101 as for ROCm 7.14.
#   ./launch.sh node6101 rocm10.1/run_rvs_node_rocm10.1.sh
set -uo pipefail
[[ -n ${_GPU_LOCKED:-} ]] || exec env _GPU_LOCKED=1 flock /dev/shm/shaohao-gpu.lock bash "$0" "$@"
export ROCM_STACK=10.1
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$BENCH_ROOT/work-rocmval"
SWEEP=$LOG_ROOT/rvs/sweep_$(date +%Y%m%d_%H%M%S)
GPU_COUNTS="1 8" DURATION_MS=30000 OUT_DIR="$SWEEP" ./run_tflops.sh
$PY analyze_rvs.py "$SWEEP" -o "$RESULTS"
sed -i 's/ROCm 7\.2\.4/ROCm 10.1.0/g' "$RESULTS/rvs_tflops.md"
echo "RVS DONE -> $RESULTS/rvs_tflops.csv"
