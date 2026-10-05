#!/usr/bin/env bash
# Wait for model staging into node-local /scratch/models (stage.log), then run Part D from
# the local copies (NFS reads of /orcd/data ran at ~8-15 MB/s during the 2026-10 reruns).
until grep -q '^Llama-3.1-70B-Instruct-FP8 done' /scratch/models/stage.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) staging complete"
export M8B=/scratch/models/Qwen3-8B-FP8 M70B=/scratch/models/Llama-3.1-70B-Instruct-FP8
exec /orcd/data/orcd/022/benchmarks/amd-ubuntu/atom/run_part_d_locked.sh "$@"
