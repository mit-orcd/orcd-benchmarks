#!/bin/bash
# Submit the B200 HPL (FP64) sweep: 1..8 GPUs on one node (one job array) and 16 GPUs on two
# nodes, all at once -> post job (analysis + push) after all of them. The launch (UCX setup) was
# checked with job-hpl-ucx-test.sh first (2 GPUs on one node, 16 GPUs on two nodes, both passed).
#   cd b200-nodes/hpl && ./submit-hpl-b200.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out-hpl
RUN=$PWD/out-hpl/run_$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"
arr=$(sbatch --parsable --array=1-8 -N 1 --ntasks-per-node=8 -J hpl-b200-g1to8 -o 'out-hpl/%x-%A_%a.out' job-hpl-b200.sh a "$RUN")
g16=$(sbatch --parsable -N 2 --ntasks-per-node=8 -J hpl-b200-g16 job-hpl-b200.sh 16 "$RUN")
post=$(sbatch --parsable --dependency=afterany:$arr:$g16 job-hpl-post.sh "$RUN")
echo "run dir: $RUN"
echo "array (1-8 GPUs): $arr  16 GPUs: $g16  post: $post"
echo "[$(date -Iseconds)] B200 HPL submitted: array $arr, g16 $g16, post $post; run $RUN" \
  >>/orcd/data/orcd/022/benchmarks/amd-ubuntu/results/status.txt
