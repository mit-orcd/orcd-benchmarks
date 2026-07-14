#!/bin/bash
# Submit the high-utilization two-node Megatron run across two B200 nodes
# (mit_testing). Same ~5B model as the 1-node MAX; inter-node gradient all-reduce
# over the B200 NDR rails. Read per-GPU TFLOP/s on the "--log-throughput" lines.
#
# Usage: ./job-megatron-2node-max.sh [nodes] [ngpus] [prec]
#   nodes: comma-separated pair (default: node5500,node5502)
#   ngpus: GPUs per node (default: 8 = full nodes); or "scan" to sweep 1..8
#   prec:  bf16 (default) or fp8 (~2x tensor-core throughput on B200)
#
# Examples:
#   ./job-megatron-2node-max.sh                          # both nodes, 8/node, bf16
#   ./job-megatron-2node-max.sh node5500,node5502 8 fp8  # both nodes, 8/node, fp8

MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
DIR=$(cd "$(dirname "$0")" && pwd)     # this script's own dir (b200-nodes)
cd "$DIR"
mkdir -p output

NODES="${1:-node5500,node5502}"
NG_ARG="${2:-8}"
PREC="${3:-bf16}"
if [ "$NG_ARG" = "scan" ]; then GPUS=(1 2 3 4 5 6 7 8); else GPUS=("$NG_ARG"); fi

for N in "${GPUS[@]}"; do
   jid=$(sbatch --parsable \
      -p mit_testing -w "$NODES" -N 2 -n 2 --exclusive \
      --gpus-per-node=b200:$N --mem=0 -t 02:00:00 \
      -J "megatron-2node-max-g$N-$PREC" \
      -o "output/megatron-2node-max-g$N-$PREC.%J" \
      --export=ALL,NG=$N,PREC=$PREC,DIR=$DIR <<'EOF'
#!/bin/bash
module load apptainer/1.4.2
MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
cd "$MEG/Megatron-LM"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
master_node=${nodes[0]}
master_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)
echo "===== nodes=${nodes[*]} master=$master_node ip=$master_ip gpus=$NG precision=$PREC ====="

srun apptainer exec \
    --nv --contain --cleanenv \
    --bind "$MEG" \
    --bind "$DIR" \
    --bind /dev/infiniband \
    --bind /sys/class/infiniband \
    --bind /sys/class/infiniband_verbs \
    "$MEG/imag/pytorch_26.02-py3.sif" \
    "$DIR/run-2nodes-b200-max.sh" "$SLURM_NNODES" "$NG" "$master_ip" "$PREC"
EOF
)
   echo "Submitted megatron 2-node MAX ($NODES), $N GPU(s)/node, $PREC: job $jid"
done
