#!/usr/bin/env bash
# GATE: cheap single-node check that the B200 nodes can run this image at all.
#
# Submit with:   sbatch job-gate-b200.sh
#
# Runs BEFORE the 1.56 TB download and before any 2-node allocation, because the one
# thing that can kill this whole plan is cheap to test: the vllm kimi-k3 image is a
# CUDA 13 (cu130) build with no cu129 tag, so the host driver must be r580 or newer.
# Twenty minutes here beats discovering it after a 1.5 TB fetch and a 2-node hold.
#
# Gates on BEHAVIOUR, not on version strings alone: real device, all 8 visible, a
# matmul that is numerically correct, and throughput above a floor. (The MI355X gate
# learned this the hard way -- torch's compiled arch list was the wrong signal there,
# because hipBLASLt carried its own tuned kernels independently of it.)
#
# Slurm targeting comes from ./notes and was verified live against `scontrol show res`:
#   reservation rres_joohye_2026-08-20_lj4j2ya3   ACTIVE 2026-08-20 -> 2026-08-27
#   Nodes=node5700-c1,node5701-c1  PartitionName=mit_testing
#   Accounts=rres_acc_joohye_2026-08-20_lj4j2ya3
# The account flag is required: the reservation is account-restricted, so a job charged
# anywhere else is refused entry to it.
# QOS is deliberately left at the default (`normal`). The reservation carries its own
# QOS `rres_qos_joohye_2026-08-20_lj4j2ya3`, but it is flagged RequiresReservation +
# OverPartQOS and exists to lift partition limits we do not hit -- mit_testing already
# allows MaxTime=7-00:00:00 and AllowQos=normal,unlimited. If Slurm ever refuses the
# submission, add: #SBATCH -q rres_qos_joohye_2026-08-20_lj4j2ya3
#SBATCH -p mit_testing
#SBATCH --reservation=rres_joohye_2026-08-20_lj4j2ya3
#SBATCH -A rres_acc_joohye_2026-08-20_lj4j2ya3
#SBATCH -w node5700-c1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=b200:8
#SBATCH --mem=200G
#SBATCH -t 00:30:00
#SBATCH -J kimi-gate
#SBATCH --exclusive
#SBATCH -o out/kimi-gate.%J.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source common/env.sh
module load "$APPTAINER_MODULE" 2>/dev/null || module load apptainer 2>/dev/null

TS=$(date +%Y%m%d_%H%M%S)
G=$LOG_ROOT/gate_$TS; mkdir -p "$G"
say() { echo "[$(date -Iseconds)] $*" | tee -a "$G/STATE.txt"; }

say "gate on $SLURMD_NODENAME (job $SLURM_JOB_ID)"
# The image is pulled exactly once by ./pull-image.sh on the login node; this job
# only ever consumes it.
check_image || { say "ABORT: image not ready -- run ./pull-image.sh first"; exit 1; }
say "image OK: $VLLM_SIF"
sed 's/^/    /' "$VLLM_SIF_MANIFEST" 2>/dev/null | tee -a "$G/STATE.txt"

# ---- staged checkpoint reachable from this node? ------------------------------------
# Metadata only; reads none of the 1.42 TiB. The login node mounts the export, which
# says nothing about this node.
if check_model; then
  say "model OK: $MKIMI -> $(readlink -f "$MKIMI")"
else
  say "ABORT: the pre-staged checkpoint is not usable from $SLURMD_NODENAME."
  say "       See ./submit.sh probe output for which nodes can see it."
  exit 1
fi

# ---- host side: driver version -----------------------------------------------------
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
say "driver=$DRV  gpus=$NGPU  mem_per_gpu=${MEM} MiB"
nvidia-smi >"$G/nvidia-smi.log" 2>&1

DRV_MAJOR=${DRV%%.*}
if [[ "${DRV_MAJOR:-0}" -lt 580 ]]; then
  say "ABORT: driver $DRV < r580. The vllm:${VLLM_TAG} image is cu130-only; there is no"
  say "       cu129 tag and the K3 wheels are not on the cu129 nightly index."
  say "       Options: ask ORCD to update the driver, or build vLLM from the K3 branch"
  say "       against cu129 torch. Do NOT start the 1.56 TB download until this is fixed."
  exit 1
fi
say "driver OK (r$DRV_MAJOR >= r580)"

# ---- the arithmetic that forces 2 nodes --------------------------------------------
# Recorded here from the live device rather than assumed, because the entire run layout
# (TP8 x PP2 instead of TP8) rests on this number.
TOT_GIB=$(python3 -c "print(f'{$MEM*$NGPU/1024:.1f}')")
say "node HBM = $NGPU x $MEM MiB = ${TOT_GIB} GiB; Kimi-K3 weights = 1454.2 GiB (1.42 TiB)"
python3 -c "
tot = $MEM * $NGPU / 1024
need = 1560998987867 / 2**30
print(('FITS' if tot > need else 'DOES NOT FIT') + f': {tot:.1f} GiB HBM vs {need:.1f} GiB of weights')
" | tee -a "$G/STATE.txt"

# ---- IB rails ----------------------------------------------------------------------
say "IB devices visible to the host:"
ibstat -l 2>/dev/null | tr '\n' ' ' | tee -a "$G/STATE.txt"; echo

# ---- container side ----------------------------------------------------------------
say "container gate: torch + vllm + a correct, fast matmul"
apptainer exec $(apt_args) "$VLLM_SIF" $PY_C -c "
import torch, time, os
print('torch', torch.__version__, 'cuda', torch.version.cuda)
try:
    import vllm; print('vllm', vllm.__version__)
except Exception as e:
    print('vllm import FAILED:', e)
p = torch.cuda.get_device_properties(0)
print('device', p.name)
print('capability %d.%d' % (p.major, p.minor))
print('devices', torch.cuda.device_count())
print('mem_gib %.1f' % (p.total_memory / 2**30))
a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
ref = a.float() @ b.float()
c = a @ b
err = (c.float() - ref).abs().max().item() / ref.abs().max().item()
print('rel_err %.3e' % err)
for _ in range(20): c = a @ b
torch.cuda.synchronize()
n = 200; t0 = time.perf_counter()
for _ in range(n): c = a @ b
torch.cuda.synchronize()
print('TFLOPS %.1f' % (2 * 4096**3 * n / (time.perf_counter() - t0) / 1e12))
# The K3 arch must actually be registered in THIS image, not just in vllm main.
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print('KimiK3_supported', 'KimiK3ForConditionalGeneration' in archs)
# Ray is bolted on from \$PYLIBS, not shipped in the image -- so prove it is importable
# here rather than discovering it when the 2-node server tries to start.
import ray
print('ray_version', ray.__version__)
from vllm.v1.executor.ray_executor import RayDistributedExecutor
print('ray_executor_ok', True)
" >"$G/gate.log" 2>&1
rc=$?
sed 's/^/    /' "$G/gate.log" | tee -a "$G/STATE.txt"

cap=$(grep -oP '^capability \K\S+'        "$G/gate.log" | head -1)
ndev=$(grep -oP '^devices \K[0-9]+'       "$G/gate.log" | head -1)
tf=$(grep -oP '^TFLOPS \K[0-9.]+'         "$G/gate.log" | head -1)
relerr=$(grep -oP '^rel_err \K\S+'        "$G/gate.log" | head -1)
k3=$(grep -oP '^KimiK3_supported \K\S+'   "$G/gate.log" | head -1)
rayv=$(grep -oP '^ray_version \K\S+'        "$G/gate.log" | head -1)
rayok=$(grep -oP '^ray_executor_ok \K\S+'   "$G/gate.log" | head -1)

ok=1
[[ $rc -ne 0 ]] && ok=0
# Blackwell B200 is sm_100. A capability that is not 10.x means the image is running
# somewhere unexpected, or the driver is presenting a fallback device.
[[ "$cap" == 10.* ]] || ok=0
[[ "${ndev:-0}" -eq 8 ]] || ok=0
[[ "$k3" == "True" ]] || ok=0
[[ "$rayok" == "True" ]] || ok=0
# 1400 TF/s floor: well under B200's ~2250 TF/s BF16 dense peak (a 4096^3 matmul does
# not reach peak), but far above anything a broken/emulated kernel path produces.
awk -v t="${tf:-0}" 'BEGIN{exit !(t>1400)}' || ok=0

if [[ $ok -ne 1 ]]; then
  say "GATE FAILED (rc=$rc cap=$cap devices=$ndev tflops=$tf rel_err=$relerr kimi_k3=$k3 ray=$rayv/$rayok)"
  exit 1
fi
say "GATE PASSED: sm_$cap, $ndev devices, ${tf} TF/s BF16, rel_err=$relerr, KimiK3 registered, ray $rayv"
say "logs: $G"
