#!/bin/bash
# Submit single-node Megatron-LM (GPT pretrain) on a B200 node, mit_testing.
# Scans GPUs-per-node 1..8 by default (one Slurm job per GPU count), or runs a
# single GPU count if given. Adapted from 1node.sh + run-1node.sh for B200.
#
# Usage: ./job-megatron-1node.sh [node] [ngpus]
#   node:  target node (default: node5500)
#   ngpus: single GPU count to run; if omitted, scan 1 2 3 4 5 6 7 8
#
# Examples:
#   ./job-megatron-1node.sh                 # node5500, scan 1..8
#   ./job-megatron-1node.sh node5502        # node5502, scan 1..8
#   ./job-megatron-1node.sh node5500 4      # node5500, just 4 GPUs
#
# Each job runs the pytorch_26.02 container (apptainer) and calls
# run-1node-b200.sh, which computes the global batch size from the GPU count.

MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
DIR=$(cd "$(dirname "$0")" && pwd)     # this script lives in $MEG/Megatron-LM
cd "$DIR"                              # so sbatch -o output/... and cwd resolve here
mkdir -p output

NODE="${1:-node5500}"
if [ -n "$2" ]; then GPUS=("$2"); else GPUS=(1 2 3 4 5 6 7 8); fi

for N in "${GPUS[@]}"; do
   jid=$(sbatch --parsable \
      -p mit_testing -w "$NODE" -N 1 -n 1 --exclusive \
      --gpus-per-node=b200:$N --mem=200GB -t 05:00:00 \
      -J "megatron-1node-$NODE-g$N" \
      -o "output/megatron-1node-$NODE-g$N.%J" \
      --export=ALL,NG=$N <<'EOF'
#!/bin/bash
module load apptainer/1.4.2
MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
cd "$MEG/Megatron-LM"
echo "===== node=$SLURMD_NODENAME gpus_per_node=$NG ====="
srun -n 1 apptainer exec \
    --nv --contain --cleanenv \
    --bind "$MEG" \
    "$MEG/imag/pytorch_26.02-py3.sif" \
    "$MEG/Megatron-LM/run-1node-b200.sh" "$NG"
EOF
)
   echo "Submitted megatron 1-node on $NODE, $N GPU(s): job $jid"
done
