# gpu-fryer

## Introduction

[gpu-fryer](https://github.com/huggingface/gpu-fryer) is a GPU stress
test from Hugging Face. It runs a sustained matrix-multiply load on each
GPU and reports achieved TFLOP/s, temperature, and HBM/thermal
throttling — useful for spotting underperforming or throttling GPUs in a
node. Runs here from an Apptainer/Singularity image.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/gpu-fryer
sbatch job.sh                          # 1 H200 GPU: fp32, bf16, fp8 for 300 s each
# when done:
grep -A3 -E '^GPU #[0-9]+: +[0-9]+ Gflops' output/<node>-<jobid>.out
grep -c 'All GPUs seem healthy' output/<node>-<jobid>.out   # 3 = all precisions passed
```

## Installation

See [install.md](install.md): pull `gpu-fryer_1.1.0.sif` into this dir.

## Usage

Each job runs gpu-fryer three times, with `--use-fp32`, `--use-bf16` and
`--use-fp8`, 300 s each (`ELAPSE` in the script), on every GPU the job holds:

```bash
singularity exec --nv -B /lib64:/home/$USER/lib64 gpu-fryer_1.1.0.sif \
    gpu-fryer --use-bf16 --nvml-lib-path /home/$USER/lib64/libnvidia-ml.so.1 300
```

The output goes to `output/<node>-<jobid>.out`.

### Automated, many runs — `submit.sh`

`submit.sh` submits `test.sh` once per node in `h200.txt` (8 × H200) and
`rtx6000.txt` (8 × RTX PRO 6000). Each job uses all 8 GPUs of its node. Edit the
node lists, then:

```bash
./submit.sh
```

`test.sh` has `-w node4900` and `-p mit_testing` in its header. The `-w` on the
`sbatch` command line overrides the node, but check the partition before you
submit.

### Single run — root dir

| Script | GPUs | Partition |
|---|---|---|
| `job.sh` | 1 × H200 (`--gres=gpu:h200:1`) | `mit_normal_gpu` |
| `l40s.sh` | 1 GPU of any type (`--gres=gpu:1`) | `mit_normal_gpu` |
| `test.sh` | 8 × H200 on one named node | `mit_testing` |

```bash
sbatch job.sh
sbatch -w node3000 --gres=gpu:l40s:4 l40s.sh    # override GPUs and node on the command line
```

L40S has no FP8 support, so on L40S the fp8 run stops with
`Error: FP8 was explicitly requested but not all GPUs support it`. The fp32 and
bf16 results are still valid.

## Analysis

At the end of each precision run, gpu-fryer prints a summary for every GPU:

```
GPU #0: 196432 Gflops/s (min: 193514.05, max: 200111.12, dev: 196432.41)
         Temperature: 63.88°C (min: 62.00, max: 64.00)
         Throttling HW: false, Thermal SW: false, Thermal HW: false
All GPUs seem healthy
```

The runs come in the order fp32, bf16, fp8, and each starts with a
`======== Run with <prec> ==========` line. To pull out the summaries:

```bash
grep -E '^=+ Run with|^GPU #[0-9]+: +[0-9]+ Gflops|Throttling HW|seem healthy|not healthy' output/<file>.out
```

A node passes when every precision ends with `All GPUs seem healthy`, all
throttling flags are `false`, and the per-GPU Gflops/s values agree within a few
percent. A GPU that is much slower than the others in its node, or reports
throttling, is suspect.
