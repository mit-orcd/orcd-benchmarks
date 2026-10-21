#!/bin/bash
# Submit the B200 HPL (FP64) sweep: smoke (1 GPU, small N) -> 1..8 GPUs on one node each and
# 16 GPUs on two nodes (all at once, after the smoke passes) -> post job (analysis + push).
# All of these go to the top of my B200 jobs (scontrol top).
#   cd b200-nodes/hpl && ./submit-hpl-b200.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out-hpl
RUN=$PWD/out-hpl/run_$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"
smoke=$(sbatch --parsable -N 1 --ntasks-per-node=1 -t 20 -J hpl-b200-smoke job-hpl-b200.sh 1 "$RUN" smoke)
ids=()
for G in 1 2 3 4 5 6 7 8; do
  ids+=("$(sbatch --parsable --dependency=afterok:$smoke --kill-on-invalid-dep=yes -N 1 --ntasks-per-node=$G -J hpl-b200-g$G job-hpl-b200.sh $G "$RUN")")
done
ids+=("$(sbatch --parsable --dependency=afterok:$smoke --kill-on-invalid-dep=yes -N 2 --ntasks-per-node=8 -J hpl-b200-g16 job-hpl-b200.sh 16 "$RUN")")
post=$(sbatch --parsable --dependency=afterany:$smoke:$(IFS=:; echo "${ids[*]}") job-hpl-post.sh "$RUN")
all="$smoke,$(IFS=,; echo "${ids[*]}")"
scontrol top "$all" || echo "WARN: scontrol top failed"
echo "run dir: $RUN"
echo "smoke: $smoke  sweep: ${ids[*]}  post: $post"
echo "[$(date -Iseconds)] B200 HPL submitted: smoke $smoke, sweep ${ids[*]}, post $post; run $RUN" \
  >>/orcd/data/orcd/022/benchmarks/amd-ubuntu/results/status.txt
