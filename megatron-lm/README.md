# megatron-lm

Single- and multi-node GPT pre-training throughput using NVIDIA's
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM). Reports iteration
time, per-GPU TFLOP/s, and loss curves.

## Prerequisites

The Megatron-LM source tree and its dependencies (Apex, custom Python
libs, container image) are **not** committed to this repo. Set them up
locally once:

```bash
# Clone upstream Megatron-LM into this directory
git clone https://github.com/NVIDIA/Megatron-LM.git

# Copy the site-specific launchers into the cloned tree
cp site-scripts/{submit.sh,run.sh,job.sh} Megatron-LM/

# Build / pull the PyTorch + Megatron container (see notes.apptainer)
# Install Apex (see notes.conda) if running bare-metal
```

The three site scripts live in `site-scripts/` so they can be committed
without pulling in the upstream Megatron-LM source.

## Run (single benchmark)

```bash
cd Megatron-LM
sbatch job.sh 128           # global batch size 128 (single-node default)
```

Outputs land in `Megatron-LM/output/`.

## Run via py-all-bench

```bash
cd ../py-all-bench
module load miniforge/24.3.0-0
python bench_submit.py  megatron-lm \
    --partition mit_normal_gpu --nodes 3506 3507 \
    --gpu-type l40s --gpus 4 --qos unlimited
python bench_analyze.py megatron-lm \
    --partition mit_normal_gpu --num-results 2
```

`py-all-bench` submits one job per GPU-count power-of-two (1, 2, 4, …)
for each node, and additionally every node pair, with `GBS = 128 × total_GPUs`.
