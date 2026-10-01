#!/bin/bash
# Submit every B300 single-node benchmark on node5900-c1 (mit_testing), plus the
# tuned max-throughput Megatron sweep on both B300 and B200, and the analysis jobs.
#
# Benchmark jobs have no dependencies between them. They are all --exclusive on
# the one B300 node, so Slurm runs them back to back there. Each benchmark
# group gets a small CPU analysis job (mit_normal, afterany) that re-runs
# analyze-all.sh as soon as that group finishes, so summaries and
# COMPARISON-b200-vs-b300.md fill in incrementally; a final analysis job runs
# after everything. Everything lives in Slurm, so it survives logout.
#
# Usage: ./submit-all.sh
DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR" || exit 1
mkdir -p slurm-logs out-ibwrite
LOG=slurm-logs/submit-all-$(date +%Y%m%d-%H%M%S).log

jobs_of () { grep -oE 'job [0-9]+' | awk '{print $2}' | paste -sd: ; }

analysis () {  # $1 = label, $2 = colon-separated job ids
   [ -z "$2" ] && { echo "no jobs for $1, skipping its analysis" | tee -a "$LOG"; return; }
   a=$(sbatch --parsable -p mit_normal -c 1 --mem=2G -t 15 \
      --dependency=afterany:$2 -J "analyze-b300-$1" \
      -o "$DIR/slurm-logs/%x-%J.out" --wrap "$DIR/analyze-all.sh")
   echo "Submitted analysis after $1 ($2): job $a" | tee -a "$LOG"
   ALL_ANALYSIS="$ALL_ANALYSIS:$2"
}

ALL_ANALYSIS=""
{ echo "probe: job $(sbatch --parsable job-probe.sh)"; } | tee -a "$LOG"

J=$(./job-gpu-fryer.sh node5900-c1 300   | tee -a "$LOG" | jobs_of); analysis gpu-fryer "$J"
J=$(./job-nccl-1node.sh node5900-c1 all  | tee -a "$LOG" | jobs_of); analysis nccl "$J"
J=$( { echo "ibwrite: job $(sbatch --parsable job-ibwrite-1node.sh)"; } | tee -a "$LOG" | jobs_of); analysis ibwrite "$J"
J=$(./job-megatron-1node.sh node5900-c1  | tee -a "$LOG" | jobs_of); analysis megatron "$J"
# tuned max-throughput sweep, separately on B300 and on B200 (any free node)
J=$( { ./job-megatron-max-sweep.sh b300; ./job-megatron-max-sweep.sh b200 any; } | tee -a "$LOG" | jobs_of)
analysis megatron-max "$J"

analysis final "${ALL_ANALYSIS#:}"
echo "Submission log: $LOG"
