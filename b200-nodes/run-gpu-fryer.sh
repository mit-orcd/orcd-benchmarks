#!/bin/bash
# gpu-fryer GPU stress test on a single B200 node.
# Runs locally on the node (no slurm) — ssh to the node first, then ./run-gpu-fryer.sh
# Usage: ./run-gpu-fryer.sh [elapse_seconds]   (default 300)
#
# Auto-adapts to per-node differences seen across the B200 nodes:
#   - singularity binary: system /usr/bin if present, else the apptainer module.
#   - SIF mount: normal squashfuse mount where FUSE is allowed; falls back to
#     --unsquash (extract to sandbox) on nodes where /dev/fuse is restricted.

GPUFRYER_DIR=/orcd/data/orcd/022/benchmarks/gpu-fryer
SIF=$GPUFRYER_DIR/gpu-fryer_1.1.0.sif
OUT_DIR=$(cd "$(dirname "$0")" && pwd)/out-gpu-fryer
mkdir -p "$OUT_DIR"

# load the apptainer module on every node (works on all B200 nodes tested)
module load apptainer/1.4.2
which singularity
singularity --version

BIND="-B /lib64:/home/$USER/lib64"
FLAGS="--nvml-lib-path /home/$USER/lib64/libnvidia-ml.so.1"
ELAPSE="${1:-300}"

# probe: does a normal FUSE mount work on this node? if not, use --unsquash
MODE=""
probe=$(singularity exec --nv $BIND "$SIF" true 2>&1)
if echo "$probe" | grep -qiE "fuse|squashfuse|mount .*proc|operation not permitted|Permission denied"; then
    MODE="--unsquash"
    echo "FUSE mount unavailable on $(hostname); using --unsquash"
fi

run_fryer() {  # $1 = precision flag
    singularity exec $MODE --nv $BIND "$SIF" gpu-fryer "$1" $FLAGS $ELAPSE
}

OUT=$OUT_DIR/gpu-fryer-$(hostname)-$(date +%Y%m%d-%H%M%S).out

echo "Node = $(hostname)"        | tee "$OUT"
nvidia-smi -L                    | tee -a "$OUT"

echo "======== Run with fp32 ==========" | tee -a "$OUT"
run_fryer --use-fp32 2>&1 | tee -a "$OUT"
echo "======== Run with bf16 ==========" | tee -a "$OUT"
run_fryer --use-bf16 2>&1 | tee -a "$OUT"
echo "======== Run with fp8  ==========" | tee -a "$OUT"
run_fryer --use-fp8  2>&1 | tee -a "$OUT"

echo "Output written to $OUT"
