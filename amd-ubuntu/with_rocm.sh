#!/usr/bin/env bash
# Run a script with a chosen host ROCm stack (see ROCM_STACK in amd-software/env.sh).
#   ./launch.sh node6100 with_rocm.sh 7.14 run_all_node.sh "A B"     # RVS + RCCL on ROCm 7.14
#   ./launch.sh node6100 with_rocm.sh 7.14 rccl-tests/run_part_b_2node.sh
# Results go to results/<node>/rocm7.14/, logs to logs/<node>/rocm7.14/.
set -uo pipefail
export ROCM_STACK=${1:?usage: $0 <host|7.14> <script> [args]}; shift
S=${1:?script}; shift
cd /orcd/data/orcd/022/benchmarks/amd-ubuntu/"$(dirname "$S")" || exit 1
exec bash /orcd/data/orcd/022/benchmarks/amd-ubuntu/"$S" "$@"
