# gpu-fryer

## Introduction

[gpu-fryer](https://github.com/huggingface/gpu-fryer) is a GPU stress
test from Hugging Face. It runs a sustained matrix-multiply load on each
GPU and reports achieved TFLOP/s, temperature, and HBM/thermal
throttling — useful for spotting underperforming or throttling GPUs in a
node. Runs here from an Apptainer/Singularity image.

## Installation

Pull the image (the `.sif` is **not** committed — it is multi-GB) and
place it in this directory:

```bash
module load apptainer
singularity pull gpu-fryer_1.1.0.sif \
    docker://ghcr.io/huggingface/gpu-fryer:1.1.0
```

The container needs NVML (`libnvidia-ml.so.1`); the job scripts
bind-mount the host `/lib64` and pass `--nvml-lib-path` so the tool can
find it.

## Usage

The tool is invoked as `gpu-fryer [--nvml-lib-path <path>] <seconds>`.

### Automated, many runs — `submit.sh`

`submit.sh` sbatches `job.sh` for a list of nodes (one job per node).
Edit the node list inside it, then:

```bash
./submit.sh
```

Output lands in `output/<node>-<jobid>.out`.

### Single run — root dir

```bash
sbatch job.sh        # one node; runs gpu-fryer for its built-in duration
# l40s.sh is a partition-specific variant of job.sh
```

For an interactive check inside the container:

```bash
singularity shell --nv -B /lib64:/home/$USER/lib64 gpu-fryer_1.1.0.sif
gpu-fryer --nvml-lib-path /home/$USER/lib64/libnvidia-ml.so.1 60
```

## Analysis

Each output file lists per-GPU achieved TFLOP/s plus temperature and
throttling flags. Compare TFLOP/s across GPUs in a node — outliers, or
GPUs reporting thermal/HBM throttling, indicate a problem.
