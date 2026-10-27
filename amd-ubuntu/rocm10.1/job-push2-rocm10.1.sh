#!/bin/bash
# Slurm fallback for push_when_done.sh (in case the login-node waiter is stopped at logout):
# if run_rocm10.1_all.sh has written DONE, sync and push; if not, resubmit itself 1 h later
# (up to 30 times). A second push after the waiter's is harmless (nothing left to commit).
#   sbatch --begin=now+6hours rocm10.1/job-push2-rocm10.1.sh
#SBATCH -p mit_quicktest
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 15
#SBATCH -J rocm10.1-push2
#SBATCH -o /orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/launch/%x-%j.out
set -uo pipefail
B=/orcd/data/orcd/022/benchmarks
DONE=$B/amd-ubuntu/logs/node6100/rocm10.1/driver/DONE2
N=${1:-0}
if [[ ! -f $DONE ]]; then
  (( N < 30 )) && sbatch --begin=now+1hour "$B/amd-ubuntu/rocm10.1/job-push2-rocm10.1.sh" $((N + 1))
  echo "[$(date -Iseconds)] DONE not there yet (try $N)"; exit 0
fi
MSG="MI355X (ROCm 10.1) vs B200: 2-node RCCL results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
$B/sync-and-push.sh "$MSG"; rc=$?
if [[ $rc -ne 0 ]]; then
  echo "push from $(hostname -s) failed (rc=$rc); retry on login009"
  ssh -o BatchMode=yes login009 "$B/sync-and-push.sh $(printf %q "$MSG")"; rc=$?
fi
echo "[$(date -Iseconds)] push rc=$rc"
