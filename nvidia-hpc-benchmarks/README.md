# nvidia-hpc-benchmarks

## Introduction

NVIDIA's [HPC-Benchmarks](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks)
container (HPL, HPL-MxP, HPCG, STREAM) run on GPUs. This setup focuses on
**HPL** (dense LU / Linpack), reporting GFLOP/s for 1, 2, 4, and 8 GPUs
so you can see GPU LU throughput and multi-GPU scaling on a node.

## Installation

Pull the container image from NGC into `run/` (the `.sif` is **not**
committed — it is multi-GB):

```bash
module load apptainer/1.1.9
singularity pull run/hpc-benchmarks_25.04.sif \
    docker://nvcr.io/nvidia/hpc-benchmarks:25.04
```

HPL input decks (`HPL-*GPU*.dat`) ship inside the image under
`/workspace/hpl-linux-x86_64/sample-dat/`.

## Usage

### Automated, many runs — `run/`

`run.sh` submits one job per node; inside the container each job runs
`execute-all.sh`, which sweeps 1→N GPUs.

```bash
cd run
# args: "<nodes>" <partition> <reservation|none> <qos> <cpu_count> <gpu_type> <gpu_count>
./run.sh "3100 3101" mit_normal_gpu none unlimited 8 l40s 4
```

`execute-h200.sh` / `execute-mig.sh` and `submit-h200.sh` /
`submit-mig.sh` are hardware-specific variants. Output lands in
`<partition>/output-<gpu_type>/`.

### Single run — `run/job.sh` (or root `run.sh` / `job.sh`)

`run/job.sh` is a self-contained sbatch script: edit the partition, GPU
type/count, then `sbatch job.sh`. It runs HPL at 1/2/4/8 GPUs via
`singularity exec --nv … /workspace/hpl.sh --dat <deck>`.

## Analysis

```bash
cd run
# get-results.sh <partition> <N> <gpu_type>
./get-results.sh mit_normal_gpu 2 l40s
```

Prints HPL performance from the `WC0` summary lines (total and per-GPU
GFLOP/s) and the `LU GFLOPS` line for the most recent N runs; higher is
better.
