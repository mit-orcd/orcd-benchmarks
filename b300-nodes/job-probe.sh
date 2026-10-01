#!/bin/bash
# Quick environment probe of the B300 node (node5900-c1, mit_testing). Run this
# FIRST: the node is EL10 (the B200 nodes are Rocky 8), so check that the
# modules, containers, GPUs and NICs the other job scripts rely on are usable.
#
# Submit with:
#     sbatch job-probe.sh
#SBATCH -p mit_testing
#SBATCH -w node5900-c1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:b300:8
#SBATCH --mem=32GB
#SBATCH -t 15
#SBATCH -J probe-b300
#SBATCH -o slurm-logs/%x-%J.out

echo "===== $(hostname)  $(date '+%Y-%m-%d %H:%M:%S') ====="
cat /etc/os-release | grep -E '^(PRETTY_NAME|VERSION_ID)='
echo "kernel  : $(uname -r)"
echo "cpu     : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
echo "driver  : $(sed -n '1p' /proc/driver/nvidia/version 2>/dev/null)"

echo; echo "---------- GPUs ----------"
nvidia-smi --query-gpu=index,name,memory.total,power.limit,clocks.max.sm,compute_cap --format=csv
echo; nvidia-smi topo -m

echo; echo "---------- NICs ----------"
ibv_devices 2>&1 | head -20
for d in /sys/class/infiniband/*; do
   n=$(basename $d)
   echo "$n  $(cat $d/ports/1/state 2>/dev/null)  $(cat $d/ports/1/rate 2>/dev/null)  $(cat $d/ports/1/link_layer 2>/dev/null)"
done
echo "peermem : $(lsmod | grep -c nvidia_peermem) module(s)"
which ib_write_bw || echo "ib_write_bw NOT found"

echo; echo "---------- modules ----------"
module load apptainer/1.4.2 && echo "apptainer/1.4.2 OK" || echo "apptainer/1.4.2 FAILED"
which singularity apptainer 2>&1
singularity --version 2>&1
module purge
module load nvhpc/26.1 && echo "nvhpc/26.1 OK" || echo "nvhpc/26.1 FAILED"
which mpirun nvcc 2>&1
nvcc --version 2>&1 | tail -2

echo; echo "---------- container smoke tests ----------"
module load apptainer/1.4.2 2>/dev/null
ls /lib64/libnvidia-ml.so.1 || echo "libnvidia-ml.so.1 not in /lib64 (gpu-fryer bind path)"
singularity exec --nv /orcd/data/orcd/022/benchmarks/megatron-lm/imag/pytorch_26.02-py3.sif \
   python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); \
x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); print('matmul ok', (x@x).float().abs().mean().item())" 2>&1 | tail -5

echo; echo "done: $(date '+%Y-%m-%d %H:%M:%S')"
