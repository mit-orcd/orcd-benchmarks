# nvidia-hpc-benchmarks

NVIDIA HPC-Benchmarks container — HPL, HPL-MxP, HPCG, STREAM, etc. Runs
from an Apptainer/Singularity image.

## Prerequisites

Download `hpc-benchmarks_25.04.sif` (or compatible release) from NVIDIA
NGC and place it in `run/`:

```bash
module load apptainer/1.1.9
singularity pull run/hpc-benchmarks_25.04.sif \
    docker://nvcr.io/nvidia/hpc-benchmarks:25.04
```

The `.sif` file is **not** committed to this repo.

## Run (single node)

```bash
cd run
sbatch run.sh             # submits a job that runs execute-all.sh in the container
```

Other helpers in `run/`:

- `execute-all.sh` — runs every benchmark in the image
- `execute-h200.sh`, `execute-mig.sh` — variants for specific hardware
- `submit.sh`, `submit-h200.sh`, `submit-mig.sh` — thin Slurm wrappers

## Analyze

```bash
cd run
./get-results.sh          # prints GFLOPs from HPL (WC0 lines) and LU output
```

## Run via py-all-bench

```bash
cd ../py-all-bench
module load miniforge/24.3.0-0
python bench_submit.py  nvidia-hpc-benchmarks \
    --partition mit_normal_gpu --nodes 3506 3507 \
    --gpu-type l40s --gpus 4 --qos unlimited
python bench_analyze.py nvidia-hpc-benchmarks \
    --partition mit_normal_gpu --gpu-type l40s --num-results 2
```
