#!/bin/bash
# Submit three-node Megatron-LM (GPT pretrain) across three B200 nodes, mit_testing.
# Scans GPUs-per-node 1..8 by default (one Slurm job per GPU count), or runs a
# single GPU count if given. Apples-to-apple with the ~7B B200 reference in
# ~/data022/aicr-benchmarks/Benchmark_WG/megatron-lm (see run-2nodes-b200.sh,
# which is node-count generic: it takes nnodes as its first argument).
#
# Weak scaling: global batch = 128 x total GPUs = 128 x 3 x gpus_per_node, so the
# per-GPU work (and the 32 gradient-accumulation steps) is identical to the
# 1-node and 2-node runs.
#
# Usage: ./job-megatron-3node.sh [nodes] [ngpus]
#   nodes: comma-separated triple (default: node5500,node5501,node5502)
#   ngpus: single GPUs-per-node count; if omitted, scan 1 2 3 4 5 6 7 8
#
# Examples:
#   ./job-megatron-3node.sh                                    # default trio, scan 1..8
#   ./job-megatron-3node.sh node5500,node5501,node5502 8       # all three, 8 GPUs/node
#
# Each job runs the pytorch_26.02 container (apptainer) on all three nodes and
# calls run-2nodes-b200.sh with nnodes=3 (reference ~7B model, c10d rendezvous,
# B200 NDR NICs). The container binds both the megatron-lm tree (for
# pretrain_gpt.py / the .sif image) and this dir (for the run script).
# Total GPUs in a job = 3 x ngpus.

MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
DIR=$(cd "$(dirname "$0")" && pwd)     # this script's own dir (b200-nodes)
cd "$DIR"                              # so sbatch -o output-megatron/... resolves here
mkdir -p output-megatron

NODES="${1:-node5500,node5501,node5502}"
if [ -n "$2" ]; then GPUS=("$2"); else GPUS=(1 2 3 4 5 6 7 8); fi

# short tag identifying the node set, e.g. -> 5500-5501-5502
TAG=$(echo "$NODES" | tr -d ' ' | sed 's/node//g; s/,/-/g')

for N in "${GPUS[@]}"; do
   jid=$(sbatch --parsable \
      -p mit_testing -w "$NODES" -N 3 -n 3 --exclusive \
      --gpus-per-node=b200:$N --mem=200GB -t 05:00:00 \
      -J "megatron-3node-$TAG-g$N" \
      -o "output-megatron/megatron-3node-$TAG-g$N.%J" \
      --export=ALL,NG=$N,DIR=$DIR <<'EOF'
#!/bin/bash
module load apptainer/1.4.2
MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
cd "$MEG/Megatron-LM"

# master node ip for the torchrun c10d rendezvous
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
master_node=${nodes[0]}
master_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address)
echo "===== nodes=${nodes[*]} master=$master_node ip=$master_ip gpus_per_node=$NG ====="

# -n = SLURM_NTASKS = 3 -> one apptainer launch per node; bind IB for inter-node
srun apptainer exec \
    --nv --contain --cleanenv \
    --bind "$MEG" \
    --bind "$DIR" \
    --bind /dev/infiniband \
    --bind /sys/class/infiniband \
    --bind /sys/class/infiniband_verbs \
    "$MEG/imag/pytorch_26.02-py3.sif" \
    "$DIR/run-2nodes-b200.sh" "$SLURM_NNODES" "$NG" "$master_ip"
EOF
)
   echo "Submitted megatron 3-node ($NODES), $N GPU(s)/node: job $jid"
done
