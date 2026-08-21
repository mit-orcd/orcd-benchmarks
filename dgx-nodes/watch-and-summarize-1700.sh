#!/bin/bash
# Wait for the node170[0,1] NCCL jobs to finish, then regenerate summary.md
# (now covering all 8 nodes). Launched detached (nohup+disown).
cd /orcd/data/orcd/022/benchmarks/dgx-nodes || exit 1

JOBIDS="20861676 20861677 20861678"

echo "$(date) waiting on jobs: $JOBIDS" >> make-summary.log
while true; do
    pending=$(squeue -h -j "$(echo $JOBIDS | tr ' ' ',')" -o "%i" 2>/dev/null)
    [ -z "$pending" ] && break
    sleep 60
done
echo "$(date) node1700/1701 jobs finished, regenerating summary.md" >> make-summary.log

/usr/bin/python3 gen-summary.py >> make-summary.log 2>&1

echo "$(date) done" >> make-summary.log
