# b200-nodes — installation

Nothing is built in this dir. The B200 scripts reuse three installs from the
sibling benchmark dirs, plus tools that ship with the node OS. Install the
three pieces below once; the full context is in each dir's own `install.md`.

## 1. NCCL tests for B200 (CUDA 13): `../nccl-tests/build-nvhpc-26.1`

Used by `job-nccl-1node.sh`, `job-nccl-2node.sh` and `job-nccl-2node-sharp*.sh`.

```bash
cd /orcd/data/orcd/022/benchmarks/nccl-tests
./build-nvhpc-26.1.sh
#   = module purge; module load nvhpc/26.1
#     make MPI=1 MPI_HOME=$NVHPC_HOME/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi \
#          CUDA_HOME=$NVHPC_HOME/cuda NCCL_HOME=$NVHPC_HOME/comm_libs/nccl \
#          NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_100,code=compute_100"
#     mv build build-nvhpc-26.1
ls build-nvhpc-26.1/sendrecv_perf
```

The SHARP jobs use the SHARP library and NCCL SHARP plugin from the same
nvhpc/26.1 HPC-X (`$HPCX/sharp`, `$HPCX/nccl_rdma_sharp_plugin`). There is
nothing extra to install, but the fabric must run a SHARP Aggregation Manager
(`sharp_am`). See `sharp.md`.

## 2. gpu-fryer image: `../gpu-fryer/gpu-fryer_1.1.0.sif`

Used by `job-gpu-fryer.sh` and `run-gpu-fryer.sh`.

```bash
cd /orcd/data/orcd/022/benchmarks/gpu-fryer
module load apptainer/1.4.2
singularity pull gpu-fryer_1.1.0.sif docker://ghcr.io/huggingface/gpu-fryer:1.1.0
```

If FUSE mounts are blocked on a node, the job falls back to `--unsquash`.

## 3. Megatron-LM tree and PyTorch image: `../megatron-lm/`

Used by `job-megatron-*.sh`. The model configs are in this dir
(`run-1node-b200.sh`, `run-2nodes-b200.sh`, `*-max.sh`). Only the source tree
and the image come from `megatron-lm`:

```bash
cd /orcd/data/orcd/022/benchmarks/megatron-lm
module load apptainer/1.4.2
mkdir -p imag
apptainer pull imag/pytorch_26.02-py3.sif docker://nvcr.io/nvidia/pytorch:26.02-py3
git clone https://github.com/NVIDIA/Megatron-LM.git
git -C Megatron-LM checkout 8a806e517      # tested commit
```

## 4. From the node image (nothing to install)

- `ib_write_bw` (perftest, CUDA-enabled, from the DOCA/MLNX-OFED stack) for
  `job-ibwrite-*.sh`. It is on the B200 compute nodes, not on the login nodes.
- `python3` for the `analyze-*.py` scripts, which use only the standard library.

## Output dirs

The job scripts create their own output dirs (`out-*`, `output-megatron`,
`slurm-logs`).
