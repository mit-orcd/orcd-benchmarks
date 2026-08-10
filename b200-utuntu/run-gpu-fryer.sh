#!/bin/bash
# gpu-fryer GPU stress test on a single B200 node (Ubuntu 24.04: node5700/node5701).
# Runs locally on the node — there is no Slurm here, so ssh to the node first,
# then ./run-gpu-fryer.sh
#
# Usage: ./run-gpu-fryer.sh [elapse_seconds]   (default 300, minimum 10)
#
# Differences from ../b200-nodes/run-gpu-fryer.sh (Rocky 8 + Slurm nodes):
#   - no `module load apptainer` — apptainer is /usr/bin/apptainer on these nodes
#   - no `-B /lib64:/home/$USER/lib64` bind: on Ubuntu /lib64 holds only the
#     loader. `--nv` injects the driver libs into the container, so point
#     gpu-fryer at /.singularity.d/libs/libnvidia-ml.so.1 instead.
#   - no scheduler => no exclusivity; the script refuses to start if another
#     user already has processes on the GPUs (override with FORCE=1).

set -u

GPUFRYER_DIR=/orcd/data/orcd/022/benchmarks/gpu-fryer
SIF=$GPUFRYER_DIR/gpu-fryer_1.1.0.sif
OUT_DIR=$(cd "$(dirname "$0")" && pwd)/out-gpu-fryer
mkdir -p "$OUT_DIR"

ELAPSE="${1:-300}"
if [ "$ELAPSE" -lt 10 ]; then
    echo "gpu-fryer requires at least 10 seconds; got $ELAPSE" >&2
    exit 1
fi

SINGULARITY=$(command -v apptainer || command -v singularity)
if [ -z "$SINGULARITY" ]; then
    echo "no apptainer/singularity on $(hostname)" >&2
    exit 1
fi
[ -f "$SIF" ] || { echo "missing container image: $SIF" >&2; exit 1; }

# NVML inside the container: --nv puts the host driver libs in /.singularity.d/libs
NVML=/.singularity.d/libs/libnvidia-ml.so.1
FLAGS="--nvml-lib-path $NVML"

# exclusivity check — no scheduler on these nodes
BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name \
                  --format=csv,noheader 2>/dev/null)
if [ -n "$BUSY" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "GPUs on $(hostname) are already in use:" >&2
    echo "$BUSY" >&2
    echo "Results would be meaningless. Re-run with FORCE=1 to override." >&2
    exit 1
fi

# probe: does a normal FUSE mount of the SIF work on this node? if not, unpack it
MODE=""
probe=$($SINGULARITY exec --nv "$SIF" true 2>&1)
if echo "$probe" | grep -qiE "fuse|squashfuse|mount .*proc|operation not permitted|Permission denied"; then
    MODE="--unsquash"
    echo "FUSE mount unavailable on $(hostname); using --unsquash"
fi

run_fryer() {  # $1 = precision flag
    $SINGULARITY exec $MODE --nv "$SIF" gpu-fryer "$1" $FLAGS "$ELAPSE"
}

OUT=$OUT_DIR/gpu-fryer-$(hostname)-$(date +%Y%m%d-%H%M%S).out

{
    echo "Node        = $(hostname)"
    echo "Date        = $(date)"
    echo "OS          = $(. /etc/os-release; echo "$PRETTY_NAME") / $(uname -r)"
    echo "Container   = $SIF"
    echo "apptainer   = $SINGULARITY ($($SINGULARITY --version))"
    echo "Duration    = $ELAPSE s per precision"
    nvidia-smi -L
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
} | tee "$OUT"

for prec in fp32 bf16 fp8; do
    echo "======== Run with $prec ==========" | tee -a "$OUT"
    run_fryer "--use-$prec" 2>&1 | tee -a "$OUT"
done

echo "Output written to $OUT"
