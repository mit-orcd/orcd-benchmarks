#!/bin/bash
# Submit the high-utilization single-node Megatron run on the B300 node (mit_testing).
# ~5B GPT, full activation recompute, large GEMMs -> high achieved TFLOP/s, sized
# to stay well inside 180 GB HBM (no OOM). Read per-GPU TFLOP/s on the
# "--log-throughput" lines in the output file.
#
# Usage: ./job-megatron-1node-max.sh [node] [ngpus] [prec]
#   node:  target node (default: node5900-c1)
#   ngpus: GPU count (default: 8 = full node); or "scan" to sweep 1..8
#   prec:  bf16 (default) or fp8 (~2x tensor-core throughput on B200)
#
# Examples:
#   ./job-megatron-1node-max.sh                    # node5900-c1, 8 GPUs, bf16
#   ./job-megatron-1node-max.sh node5900-c1 8 fp8     # node5900-c1, 8 GPUs, fp8
#   ./job-megatron-1node-max.sh node5900-c1 scan      # node5900-c1, sweep 1..8, bf16
#   GPU_TYPE=b200 ./job-megatron-1node-max.sh any 8   # B200 baseline, any free B200 node
#
# GPU_TYPE (env, default b300) selects the GPU gres; node "any" lets Slurm pick
# the node, and the output file is then named after the node it ran on.

MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
DIR=$(cd "$(dirname "$0")" && pwd)     # this script's own dir (b300-nodes)
cd "$DIR"
mkdir -p output

NODE="${1:-node5900-c1}"
NG_ARG="${2:-8}"
PREC="${3:-bf16}"
if [ "$NG_ARG" = "scan" ]; then GPUS=(1 2 3 4 5 6 7 8); else GPUS=("$NG_ARG"); fi

GPU_TYPE="${GPU_TYPE:-b300}"
if [ "$NODE" = "any" ]; then NODE_OPT=(); TAG=%N; else NODE_OPT=(-w "$NODE"); TAG=$NODE; fi

for N in "${GPUS[@]}"; do
   jid=$(sbatch --parsable \
      -p mit_testing "${NODE_OPT[@]}" -N 1 -n 1 --exclusive \
      --gpus-per-node=$GPU_TYPE:$N --mem=0 -t 02:00:00 \
      -J "megatron-1node-max-$NODE-g$N-$PREC" \
      -o "output/megatron-1node-max-$TAG-g$N-$PREC.%J" \
      --export=ALL,NG=$N,PREC=$PREC,DIR=$DIR <<'EOF'
#!/bin/bash
module load apptainer/1.4.2 2>/dev/null || echo "apptainer module failed; using system binary"
MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
cd "$MEG/Megatron-LM"
echo "===== node=$SLURMD_NODENAME gpus=$NG precision=$PREC ====="
srun -n 1 apptainer exec \
    --nv --contain --cleanenv \
    --bind "$MEG" \
    --bind "$DIR" \
    "$MEG/imag/pytorch_26.02-py3.sif" \
    "$DIR/run-1node-b300-max.sh" "$NG" "$PREC"
EOF
)
   echo "Submitted megatron 1-node MAX ($GPU_TYPE) on $NODE, $N GPU(s), $PREC: job $jid"
done
