#!/bin/bash
# Post-run analysis for the node5700-c1 / node5701-c1 re-test (gpu-fryer,
# nccl 1-node, nccl 2-node). Submit with --dependency=afterany:<test jobs>.
#
# - out-nccl-1node/summary.md is fully generated, so it is regenerated in place
#   (newest file per node wins, so the two nodes are added as new columns).
# - out-gpu-fryer/summary.md and out-nccl-2node/summary.md carry hand-written
#   "New nodes" sections that a regeneration would erase, so the analyzers are
#   pointed at a staging dir holding only the new runs; the result is merged into
#   the main summaries by hand.
#SBATCH -p mit_quicktest
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 15
#SBATCH -J analyze-5700-5701
#SBATCH -o slurm-logs/%x-%J.out

DIR=/orcd/data/orcd/022/benchmarks/b200-nodes
cd "$DIR" || exit 1

python3 analyze-nccl-1node.py > /dev/null && echo "nccl-1node summary regenerated"

stage() {   # stage <analyzer> <out dir> <file glob>
   local s="${2:?}/staged-5700-5701"
   rm -rf "$s"; mkdir -p "$s"
   for f in $3; do [ -f "$f" ] && ln -s "$DIR/$f" "$s/"; done
   python3 - "$1" "$s" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("a", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.OUT_DIR = sys.argv[2]; sys.argv = [sys.argv[1]]
m.main()
PY
}
stage analyze-gpu-fryer.py out-gpu-fryer "out-gpu-fryer/gpu-fryer-node5700-c1-*.out out-gpu-fryer/gpu-fryer-node5701-c1-*.out" > /dev/null && echo "gpu-fryer staged"
stage analyze-nccl-2node.py out-nccl-2node "out-nccl-2node/nccl-2node-5700-c1-5701-c1-*.out" > /dev/null && echo "nccl-2node staged"
