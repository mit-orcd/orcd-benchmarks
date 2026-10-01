#!/bin/bash
# Re-run every analyzer on whatever results exist so far, then rebuild the
# B200 vs B300 comparison. Safe to run at any time (missing data -> "—").
# Submitted automatically by submit-all.sh after each benchmark finishes.
cd "$(dirname "$0")" || exit 1
echo "===== analyze-all $(date '+%Y-%m-%d %H:%M:%S') on $(hostname) ====="
ls out-gpu-fryer/*.out   >/dev/null 2>&1 && python3 analyze-gpu-fryer.py  >/dev/null && echo "gpu-fryer summary ok"
ls out-nccl-1node/*.out  >/dev/null 2>&1 && python3 analyze-nccl-1node.py >/dev/null && echo "nccl summary ok"
ls output-megatron/megatron-1node-* >/dev/null 2>&1 && python3 analyze-megatron.py >/dev/null && echo "megatron summary ok"
python3 compare-b200-b300.py
