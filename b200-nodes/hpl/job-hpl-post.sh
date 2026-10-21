#!/bin/bash
# After all B200 HPL jobs (afterany): analysis -> ../results/hpl-b200.md and section 2 of
# ../../amd-ubuntu/results/mi355x-vs-b200.md, then sync and push.
#   sbatch --dependency=afterany:<ids> job-hpl-post.sh <run dir>
#SBATCH -p mit_quicktest
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 15
#SBATCH -J hpl-b200-post
#SBATCH -o out-hpl/%x-%j.out
set -uo pipefail
RUN=${1:?run dir}
B=/orcd/data/orcd/022/benchmarks
mkdir -p $B/amd-ubuntu/logs/auto
if [[ ! -s $RUN/summary.txt ]]; then
  echo "no B200 HPL results in $RUN (see $RUN/smoke.txt and out-hpl/)"
  echo "[$(date -Iseconds)] B200 HPL: no results in $RUN; nothing pushed" >>$B/amd-ubuntu/results/status.txt
  exit 1
fi
flock $B/amd-ubuntu/logs/auto/report.lock python3 $B/b200-nodes/hpl/analyze-hpl-b200.py "$RUN" || exit 1
MSG="HPL FP64: B200 (NVIDIA HPL) vs MI355X (rocHPL), new section 2 of mi355x-vs-b200.md

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
$B/sync-and-push.sh "$MSG" >"$RUN/push.log" 2>&1
rc=$?
if (( rc != 0 )) || ! grep -q '==> pushed' "$RUN/push.log"; then
  # compute nodes may lack the git login; retry from the login node
  echo "push from $(hostname -s) failed (rc=$rc); retry on login009" >>"$RUN/push.log"
  ssh -o BatchMode=yes login009 "$B/sync-and-push.sh $(printf %q "$MSG")" >>"$RUN/push.log" 2>&1
  rc=$?
fi
tail -3 "$RUN/push.log"
echo "[$(date -Iseconds)] B200 HPL analysis done ($RUN), push rc=$rc" >>$B/amd-ubuntu/results/status.txt
