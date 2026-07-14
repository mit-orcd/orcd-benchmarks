#!/bin/bash
# Submit two-node Megatron-LM (GPT pretrain) across two B200 nodes, mit_testing.
# Scans GPUs-per-node 1..8 by default (one Slurm job per GPU count), or runs a
# single GPU count if given. Apples-to-apple with the ~7B B200 reference in
# ~/data022/aicr-benchmarks/Benchmark_WG/megatron-lm (see run-2nodes-b200.sh).
#
# Usage: ./job-megatron-2node.sh [nodes] [ngpus]
#   nodes: comma-separated pair (default: node5500,node5502)
#   ngpus: single GPUs-per-node count; if omitted, scan 1 2 3 4 5 6 7 8
#
# Examples:
#   ./job-megatron-2node.sh                          # node5500,node5502, scan 1..8
#   ./job-megatron-2node.sh node5500,node5502 8      # both nodes, 8 GPUs/node
#
# Each job runs the pytorch_26.02 container (apptainer) on both nodes and calls
# run-2nodes-b200.sh (in this dir; reference ~7B model, c10d rendezvous, B200 NDR
# NICs, global batch = 128 x total GPUs). The container binds both the megatron-lm
# tree (for pretrain_gpt.py / the .sif image) and this dir (for the run script).
# Total GPUs in a job = 2 x ngpus.

MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
DIR=$(cd "$(dirname "$0")" && pwd)     # this script's own dir (b200-nodes)
cd "$DIR"                              # so sbatch -o output/... and cwd resolve here
mkdir -p output

NODES="${1:-node5500,node5502}"
if [ -n "$2" ]; then GPUS=("$2"); else GPUS=(1 2 3 4 5 6 7 8); fi

for N in "${GPUS[@]}"; do
   jid=$(sbatch --parsable \
      -p mit_testing -w "$NODES" -N 2 -n 2 --exclusive \
      --gpus-per-node=b200:$N --mem=200GB -t 05:00:00 \
      -J "megatron-2node-g$N" \
      -o "output/megatron-2node-g$N.%J" \
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

# -n = SLURM_NTASKS = 2 -> one apptainer launch per node; bind IB for inter-node
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
   echo "Submitted megatron 2-node ($NODES), $N GPU(s)/node: job $jid"
done
