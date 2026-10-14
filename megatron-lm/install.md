# megatron-lm — installation

You need three things: the NVIDIA PyTorch container, which includes
Megatron-Core and Transformer Engine; the Megatron-LM source; and the site
launchers from `site-scripts/`. The runs use `--mock-data` with
`NullTokenizer`, so no dataset, vocab file or extra Python packages are needed.

- Source: <https://github.com/NVIDIA/Megatron-LM>
- Container: `docker://nvcr.io/nvidia/pytorch:26.02-py3`
- Docs: <https://docs.nvidia.com/megatron-core/developer-guide/latest/>

```bash
cd /orcd/data/orcd/022/benchmarks/megatron-lm
module load apptainer/1.4.2

# 1. Container image (~20 GB). job.sh expects imag/pytorch_26.02-py3.sif
mkdir -p imag
apptainer pull imag/pytorch_26.02-py3.sif docker://nvcr.io/nvidia/pytorch:26.02-py3

# 2. Megatron-LM source (tested at commit 8a806e517, 2026-03-13)
git clone https://github.com/NVIDIA/Megatron-LM.git
git -C Megatron-LM checkout 8a806e517     # optional: pin the tested version

# 3. Site launchers and the output dir
cp site-scripts/{submit.sh,run.sh,job.sh} Megatron-LM/
chmod +x Megatron-LM/run.sh
mkdir -p Megatron-LM/output
```

## Check

```bash
apptainer exec --nv --contain imag/pytorch_26.02-py3.sif \
    python -c "import megatron.core, transformer_engine.pytorch; print('ok')"
```

Run this on a GPU node, e.g. inside `salloc --gpus=1`.

## Notes

- `job.sh` and `run.sh` hard-code `work_path=/orcd/data/orcd/022/benchmarks/megatron-lm`.
  If you install somewhere else, edit both files.
- `run.sh` sets `NCCL_IB_HCA=mlx5_0..mlx5_8` for 2-node runs. Change it if
  the node has different HCA names (`ibv_devices`).
- Older images (`pytorch_24.01`, `24.07`) are still in `imag/`, but the
  current scripts use only `26.02`.
