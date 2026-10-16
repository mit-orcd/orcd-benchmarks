#!/bin/bash
# After job-kimi-mtp-b200.sh ends: put its result into results/kimi-k3-mtp-b200.md and
# ../amd-ubuntu/results/mi355x-vs-b200-kimi-k3.md (adds "Ours B200 +MTP"), then sync and push.
#   sbatch --dependency=afterany:<mtp job> --export=ALL,MTPJOB=<mtp job> job-kimi-mtp-post.sh
#SBATCH -p mit_quicktest
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 15
#SBATCH -J kimi-mtp-post
#SBATCH -o out/%x-%J.out
B=/orcd/data/orcd/022/benchmarks
ST=$B/amd-ubuntu/results/status.txt
RUN=$(cat $B/b200-kimi/out/mtp-run-$MTPJOB.txt 2>/dev/null)
if [[ -z $RUN || ! -d $RUN ]]; then
  echo "$(date -Iseconds) B200 Kimi-K3 MTP: job $MTPJOB left no run dir, md not updated" | tee -a $ST; exit 1
fi
mkdir -p $B/amd-ubuntu/logs/auto
flock $B/amd-ubuntu/logs/auto/report.lock python3 $B/b200-kimi/analyze-mtp-b200.py "$RUN" "$MTPJOB" || {
  echo "$(date -Iseconds) B200 Kimi-K3 MTP: analysis failed for job $MTPJOB ($RUN)" | tee -a $ST; exit 1; }
$B/sync-and-push.sh "Kimi-K3 InferenceX cross-check: add B200 with MTP (job $MTPJOB)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
