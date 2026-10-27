#!/usr/bin/env bash
# ROCm 10.1, this node: RCCL 1-node bandwidth at 8 GPUs (all collectives, default settings, same as
# run-rccl-all.sh) and the 1-node latency run, so MI355X interconnect numbers can be the mean of
# node6100 and node6101 as for ROCm 7.14.
#   ./launch.sh node6101 rocm10.1/run_1node_extra_rocm10.1.sh
set -uo pipefail
[[ -n ${_GPU_LOCKED:-} ]] || exec env _GPU_LOCKED=1 flock /dev/shm/shaohao-gpu.lock bash "$0" "$@"
export ROCM_STACK=10.1 GPU_LOCKS_HELD=1
source /orcd/data/orcd/022/benchmarks/amd-ubuntu/common/env.sh
cd "$BENCH_ROOT/rccl-tests"
sleep 30
GPU_COUNTS=8 OUT_TAG=all ./run-rccl-all.sh
d=$(ls -dt "$LOG_ROOT"/rccl/rccl_all_* | head -1)
$PY analyze_rccl.py "$d" -o "$RESULTS" && echo "bandwidth -> $RESULTS/rccl.csv"
sleep 30
bash "$BENCH_ROOT/rocm10.1/run_rccl_lat_1node_rocm10.1.sh"
echo "1node extra DONE"
