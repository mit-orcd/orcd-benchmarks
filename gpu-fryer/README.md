# gpu-fryer

GPU stress test that sweeps fp32 / bf16 / fp8 and reports memory usage.
Runs from an Apptainer/Singularity image.

## Prerequisites

Download `gpu-fryer_1.1.0.sif` (not committed to this repo) and place it
in this directory:

```bash
module load apptainer/1.4.2
singularity pull gpu-fryer_1.1.0.sif docker://<source>/gpu-fryer:1.1.0
```

Also ensure `/lib64/libnvidia-ml.so.1` (NVML) is accessible — the job
scripts bind-mount `/lib64` into the container.

## Run (single benchmark)

```bash
sbatch job.sh           # one node, runs fp32 + bf16 + fp8 for 300s each
# or
./submit.sh             # wrapper that sbatches job.sh for a list of nodes
```

Outputs land in `output/<node>-<jobid>.out`.

## Run via py-all-bench

```bash
cd ../py-all-bench
module load miniforge/24.3.0-0
python bench_submit.py  gpu-fryer \
    --partition mit_normal_gpu --nodes 3506 3507 \
    --gpu-type l40s --gpus 4 --qos unlimited
python bench_analyze.py gpu-fryer --partition mit_normal_gpu --num-results 2
```
