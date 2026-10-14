# b300-nodes — installation

Nothing is built in this dir. Like `../b200-nodes`, the B300 scripts reuse three
installs from the sibling benchmark dirs. If `b200-nodes` already works, there
is nothing more to do.

## 1. NCCL tests: `../nccl-tests/build-nvhpc-26.1`

```bash
cd /orcd/data/orcd/022/benchmarks/nccl-tests
./build-nvhpc-26.1.sh       # module nvhpc/26.1 (CUDA 13 + HPC-X Open MPI + NCCL); sm_90, sm_100 + compute_100 PTX
```

The build has no `sm_103` code. B300 runs the `compute_100` PTX through the
driver JIT, so no separate B300 build is needed. `run-nccl-1node.sh` loads
`nvhpc/26.1` and exits if the module does not load on the node.

## 2. gpu-fryer image: `../gpu-fryer/gpu-fryer_1.1.0.sif`

```bash
cd /orcd/data/orcd/022/benchmarks/gpu-fryer
module load apptainer/1.4.2
singularity pull gpu-fryer_1.1.0.sif docker://ghcr.io/huggingface/gpu-fryer:1.1.0
```

## 3. Megatron-LM tree and PyTorch image: `../megatron-lm/`

The B300 model configs are in this dir (`run-1node-b300.sh`, `*-max*.sh`).

```bash
cd /orcd/data/orcd/022/benchmarks/megatron-lm
module load apptainer/1.4.2
mkdir -p imag
apptainer pull imag/pytorch_26.02-py3.sif docker://nvcr.io/nvidia/pytorch:26.02-py3
git clone https://github.com/NVIDIA/Megatron-LM.git
git -C Megatron-LM checkout 8a806e517
```

## Node notes (EL10)

- The B300 node (node5900) runs EL10. `module load apptainer/1.4.2` may fail
  there, in which case the scripts fall back to the system
  `apptainer`/`singularity`. gpu-fryer falls back to `--unsquash` where FUSE is
  not allowed.
- `job-probe.sh` checks which modules and container runtimes work on a node
  before a full run.
- `ib_write_bw` (`job-ibwrite-1node.sh`) comes from the node's OFED/DOCA perftest.
- `analyze-*.py` and `compare-b200-b300.py` use only the Python standard library.
