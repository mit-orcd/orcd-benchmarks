#!/usr/bin/env bash
# Report driver / CUDA / GPU memory on each B200 node, so the r580+ question is settled
# before anything expensive is committed.
#
# Usage:  ./job-probe-drivers.sh [node[,node...]]
#         (default: the reserved pair AND the Rocky fallback pair -- one job per node)
#
# WHY THIS EXISTS: the vllm:kimi-k3 image is a CUDA 13 (cu130) build with no cu129 tag
# and needs an r580+ host driver. ../b200-ubuntu/ubuntu-nccl.md recorded the RESERVED
# Ubuntu pair (node570[0-1]-c1) on driver 570.211.01 / CUDA 12.9 on 2026-08-12, and the
# Rocky nodes (node550[0-2]-c1) on 590.48.01. If that is still true, this benchmark
# cannot run on the reserved nodes at all and must move to the Rocky ones.
#
# Each job holds ONE GPU for well under a minute. No image, no weights, no allocation
# of consequence -- this is the cheapest possible way to answer the question.
set -uo pipefail
cd "$(dirname "$0")"
source common/env.sh

TARGETS="${1:-$B200_NODES,$B200_NODES_ALT}"
mkdir -p "$OUT_DIR"

for NODE in ${TARGETS//,/ }; do
  # The reservation only covers its own nodes; asking for it on a node outside it is a
  # submission error, so scope the flag to the nodes the reservation actually holds.
  EXTRA=(-p "$SLURM_PART")
  if [[ ",$B200_NODES," == *",$NODE,"* && -n "${RESV:-}" ]]; then
    EXTRA+=(--reservation="$RESV")
    [[ -n "${ACCT:-}" ]] && EXTRA+=(-A "$ACCT")
  fi

  jid=$(sbatch --parsable "${EXTRA[@]}" \
        --export=ALL,MKIMI="$MKIMI",MODEL_STORE="$MODEL_STORE",KIMI_SHARDS="$KIMI_SHARDS" \
        -w "$NODE" -N 1 -n 1 --gpus-per-node=b200:1 --mem=8G -t 00:05:00 \
        -J "drv-$NODE" -o "$OUT_DIR/drv-$NODE.%J.out" <<'SBEOF'
#!/bin/bash
echo "===== $SLURMD_NODENAME ====="
echo "-- os --"
( . /etc/os-release && echo "$PRETTY_NAME" ) 2>/dev/null
uname -r
echo "-- gpu --"
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
           --format=csv,noheader | head -8
echo "-- cuda runtime advertised by the driver --"
nvidia-smi | awk '/CUDA Version/ {print}' | head -1
echo "-- staged checkpoint visible from this node? --"
# The second question this probe answers. Kimi-K3 lives on fstor025.ib:/compute/orcd/025,
# which the LOGIN node mounts -- that says nothing about the compute nodes. If they do
# not mount it, the benchmark cannot read its weights and that has to surface now, not
# 20 minutes into a 2-node allocation. Metadata only; not a byte of the 1.42 TiB is read.
if [ -d "$MKIMI" ]; then
   N=$(ls -1 "$MKIMI"/*.safetensors 2>/dev/null | wc -l)
   B=$(ls -lL "$MKIMI"/*.safetensors 2>/dev/null | awk '{t+=$5} END{print t+0}')
   echo "MODEL VISIBLE: $MKIMI"
   echo "  -> $(readlink -f "$MKIMI")"
   echo "  $N/$KIMI_SHARDS shards, $B bytes"
   [ "$N" -eq "${KIMI_SHARDS:-96}" ] && echo "  shard count OK" || echo "  SHARD COUNT WRONG"
else
   echo "MODEL NOT VISIBLE: $MKIMI is not readable from this node."
   echo "  $MODEL_STORE is not mounted here. Either ask ORCD to mount"
   echo "  fstor025.ib:/compute/orcd/025 on the B200 nodes, or stage the weights"
   echo "  somewhere these nodes can reach. The benchmark cannot run without it."
fi

echo "-- verdict --"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
MAJ=${DRV%%.*}
MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
# GPUs PER NODE, not GPUs visible to this job. The probe deliberately requests a single
# GPU to stay cheap, so nvidia-smi shows 1 -- using that count would understate node HBM
# eightfold and report a nonsense shortfall. Take the real figure from the node's Gres.
N=$(scontrol show node "$SLURMD_NODENAME" 2>/dev/null \
    | grep -oP 'Gres=gpu:[a-z0-9]+:\K[0-9]+' | head -1)
N=${N:-8}
echo "  (GPUs visible to this probe: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l); GPUs per node from Gres: $N)"
if [ "${MAJ:-0}" -ge 580 ]; then
   echo "DRIVER OK: $DRV (r$MAJ >= r580) -- the cu130 vllm:kimi-k3 image can run here"
else
   echo "DRIVER TOO OLD: $DRV (r$MAJ < r580)"
   echo "  The vllm:kimi-k3 image is cu130-only; there is no cu129 tag and the K3"
   echo "  wheels are not on the cu129 nightly index. This node cannot run it."
fi
python3 - "$MIB" "$N" <<'PY'
import sys
mib, n = int(sys.argv[1]), int(sys.argv[2])
node = mib * n * 2**20
need = 1560998987867
print(f"HBM: {n} x {mib} MiB = {node/1e9:.0f} GB per node")
print(f"Kimi-K3 weights: {need/1e9:.0f} GB")
print(("FITS ON ONE NODE by %.0f GB (weights only; KV pool, workspace and NCCL "
       "buffers still have to come out of it)" % ((node-need)/1e9)) if node > need
      else ("DOES NOT FIT ON ONE NODE: short by %.0f GB on weights alone, before KV "
            "cache, workspace or NCCL buffers -> TP8 x PP2 across two nodes" % ((need-node)/1e9)))
PY
SBEOF
)
  echo "submitted driver probe on $NODE: job $jid  -> $OUT_DIR/drv-$NODE.$jid.out"
done

echo
echo "When they finish:"
echo "  grep -h -E 'DRIVER (OK|TOO OLD)|MODEL (VISIBLE|NOT VISIBLE)|FITS|DOES NOT FIT' $OUT_DIR/drv-*.out"
