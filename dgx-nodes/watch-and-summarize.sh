#!/bin/bash
# Wait for the run-nccl.sh jobs (new build-songhan-utuntu-nvhpc-26.1 build) to
# finish, then write summary.md. Launched detached (nohup+disown) so it
# survives the interactive session logging out.
cd /orcd/data/orcd/022/benchmarks/dgx-nodes || exit 1

JOBIDS="20878187 20878188 20878189 20878190 20878191 20878192 20878193 20878194 20878195"

echo "$(date) waiting on jobs: $JOBIDS" >> make-summary.log
while true; do
    pending=$(squeue -h -j "$(echo $JOBIDS | tr ' ' ',')" -o "%i" 2>/dev/null)
    [ -z "$pending" ] && break
    sleep 60
done
echo "$(date) all jobs finished, generating summary.md" >> make-summary.log

/usr/bin/python3 gen-summary.py >> make-summary.log 2>&1

echo "$(date) done" >> make-summary.log
