#!/bin/bash
# Submit the tuned max-throughput Megatron sweep (run-1node-max-tuned.sh), one
# 8-GPU single-node job per grid point, mit_testing. No dependencies: Slurm runs
# them as nodes free up (B300 points serialize on the one B300 node).
#
# Usage: ./job-megatron-max-sweep.sh <gpu_type> [node]
#   gpu_type: b300 or b200
#   node:     target node, or "any" (default: node5900-c1 for b300, any for b200)
#
# Grid (same for both GPU types; the best point per type is what gets compared):
#   5b  : micro 4 8 16      x recompute none selective full x bf16 fp8
#   13b : micro 2 4 8       x recompute none full           x bf16 fp8
# Output: output-max-sweep/max-<gpu>-<node>-<model>-mb<m>-<recompute>-<prec>.<jobid>
#
# Prints "job <id>" lines (used by submit-all.sh / for dependencies).

MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR" || exit 1
mkdir -p output-max-sweep

GPU_TYPE="${1:?usage: $0 b300|b200 [node]}"
if [ "$GPU_TYPE" = "b300" ]; then NODE="${2:-node5900-c1}"; else NODE="${2:-any}"; fi
if [ "$NODE" = "any" ]; then NODE_OPT=(-x "${EXCLUDE:-node5800-c1}"); TAG=%N;  # node5800-c1: jobs fail to launch (2026-10-04)
   else NODE_OPT=(-w "$NODE"); TAG=$NODE; fi

submit () {  # model micro recompute prec
   local model=$1 mb=$2 rc=$3 prec=$4
   local name="max-$GPU_TYPE-$model-mb$mb-$rc-$prec"
   # skip grid points that already finished all 20 iterations (any node)
   if grep -qlE 'iteration +20/' output-max-sweep/max-$GPU_TYPE-*-$model-mb$mb-$rc-$prec.* 2>/dev/null; then
      echo "Skipped $name: already done"; return
   fi
   [ -n "$ONLY" ] && [ "$ONLY" != "$model-mb$mb-$rc-$prec" ] && return   # ONLY=5b-mb4-none-bf16 for a test
   jid=$(sbatch --parsable \
      -p mit_testing "${NODE_OPT[@]}" -N 1 -n 1 --exclusive \
      --gpus-per-node=$GPU_TYPE:8 --mem=0 -t 00:45:00 \
      -J "$name" \
      -o "output-max-sweep/max-$GPU_TYPE-$TAG-$model-mb$mb-$rc-$prec.%J" \
      --export=ALL,DIR=$DIR,ARGS="8 $model $mb $rc $prec" <<'EOF'
#!/bin/bash
module load apptainer/1.4.2 2>/dev/null || echo "apptainer module failed; using system binary"
MEG=/orcd/data/orcd/022/benchmarks/megatron-lm
cd "$MEG/Megatron-LM"
echo "===== node=$SLURMD_NODENAME args=$ARGS ====="
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
    "$DIR/run-1node-max-tuned.sh" $ARGS
rm -rf "$SCR"
# analyze as soon as this job ends (flock: jobs may finish together)
flock "$DIR/.analyze.lock" "$DIR/analyze-all.sh" > /dev/null 2>&1
EOF
)
   echo "Submitted $name on $NODE: job $jid"
}

for prec in bf16 fp8; do
   for mb in 4 8 16; do for rc in none selective full; do submit 5b  $mb $rc $prec; done; done
   for mb in 2 4 8;  do for rc in none full;           do submit 13b $mb $rc $prec; done; done
done
