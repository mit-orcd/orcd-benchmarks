#!/bin/bash
# After job-nccl-lat-2node-tune.sh ends: put its result into amd-ubuntu/results/
# mi355x-vs-b200-interconnect.md (§2.3 + Quick summary) and mi355x-vs-b200.md (§3), then sync and push.
#   sbatch --dependency=afterany:<tune job> --export=ALL,TUNEJOB=<tune job> job-nccl-lat-2node-tune-post.sh
#SBATCH -p mit_quicktest
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 15
#SBATCH -J nccl-lat-tune-post
#SBATCH -o slurm-logs/%x-%J.out
B=/orcd/data/orcd/022/benchmarks
ST=$B/amd-ubuntu/results/status.txt
D=$B/b200-nodes/out-nccl-lat-tune/$TUNEJOB
if ! grep -q "latency run DONE" "$D/STATE.txt" 2>/dev/null; then
  echo "$(date -Iseconds) B200 NCCL 1 MB tuning: job $TUNEJOB did not finish, md not updated (see $D)" | tee -a $ST; exit 1
fi
mkdir -p $B/amd-ubuntu/logs/auto
flock $B/amd-ubuntu/logs/auto/report.lock python3 $B/amd-ubuntu/rccl-tests/b200_lat_tune.py "$D" || exit 1
echo "$(date -Iseconds) B200 NCCL 1 MB tuning: job $TUNEJOB result put into mi355x-vs-b200-interconnect.md + mi355x-vs-b200.md" >> $ST
$B/sync-and-push.sh "Interconnect: B200 2-node latency with fixed NCCL_ALGO/NCCL_PROTO (job $TUNEJOB)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
