#!/bin/bash
# Submit single-node Megatron-LM (GPT pretrain) on the B300 node, mit_testing.
# Scans GPUs-per-node 1..8 by default (one Slurm job per GPU count), or runs a
# single GPU count if given. Apples-to-apple with the ~7B B200 reference in
# ~/data022/aicr-benchmarks/Benchmark_WG/megatron-lm (see run-1node-b300.sh).
#
# Usage: ./job-megatron-1node.sh [node] [ngpus]
#   node:  target node (default: node5900-c1)
#   ngpus: single GPU count to run; if omitted, scan 1 2 3 4 5 6 7 8
#
# Examples:
#   ./job-megatron-1node.sh                 # node5900-c1, scan 1..8
#   ./job-megatron-1node.sh node5900-c1        # node5900-c1, scan 1..8
#   ./job-megatron-1node.sh node5900-c1 4      # node5900-c1, just 4 GPUs
#
# Each job runs the pytorch_26.02 container (apptainer) and calls
# run-1node-b300.sh (in this dir), which holds the reference ~7B model config
# (global batch = 128 x GPUs). The container binds both the megatron-lm tree
# (for pretrain_gpt.py / the .sif image) and this dir (for the run script).

MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
DIR=$(cd "$(dirname "$0")" && pwd)     # this script's own dir (b300-nodes)
cd "$DIR"                              # so sbatch -o output-megatron/... and cwd resolve here
mkdir -p output-megatron

NODE="${1:-node5900-c1}"
if [ -n "$2" ]; then GPUS=("$2"); else GPUS=(1 2 3 4 5 6 7 8); fi

for N in "${GPUS[@]}"; do
   jid=$(sbatch --parsable \
      -p mit_testing -w "$NODE" -N 1 -n 1 --exclusive \
      --gpus-per-node=b300:$N --mem=200GB -t 05:00:00 \
      -J "megatron-1node-$NODE-g$N" \
      -o "output-megatron/megatron-1node-$NODE-g$N.%J" \
      --export=ALL,NG=$N,DIR=$DIR <<'EOF'
#!/bin/bash
module load apptainer/1.4.2 2>/dev/null || echo "apptainer module failed; using system binary"
MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
cd "$MEG/Megatron-LM"
echo "===== node=$SLURMD_NODENAME gpus_per_node=$NG ====="
# All temp files (container /tmp, torch inductor / triton caches) go to the
# user scratch dir, never /tmp: the B300 node's container /tmp filled up
# ("No space left on device") during torch.compile in the first runs.
SCR=$(readlink -f "$HOME/orcd/scratch")/tmp/megatron-$SLURM_JOB_ID
mkdir -p "$SCR/tmp" "$SCR/apptainer"
export APPTAINER_TMPDIR=$SCR/apptainer APPTAINERENV_SCR=$SCR
srun -n 1 apptainer exec \
    --nv --contain --cleanenv \
    --workdir "$SCR" \
    --bind "$MEG" \
    --bind "$DIR" \
    --bind "$SCR" \
    "$MEG/imag/pytorch_26.02-py3.sif" \
    "$DIR/run-1node-b300.sh" "$NG"
rm -rf "$SCR"
# analyze as soon as this job ends (flock: jobs may finish together)
flock "$DIR/.analyze.lock" "$DIR/analyze-all.sh" > /dev/null 2>&1
EOF
)
   echo "Submitted megatron 1-node on $NODE, $N GPU(s): job $jid"
done
