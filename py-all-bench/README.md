# py-all-bench

Python automation for submitting and analyzing HPC benchmark jobs on a Slurm
cluster. Replaces the hand-written shell scripts in `../all-bench/` and the
per-benchmark `run/run.sh` and `run/get-results.sh` scripts.

## Files

| File | Purpose |
|------|---------|
| `bench_submit.py`  | Submit Slurm benchmark jobs |
| `bench_analyze.py` | Parse output files and print key metrics |
| `bench_config.py`  | Shared paths, benchmark registry, Slurm query helpers |

## Supported benchmarks

| Name                     | Category | Needs GPU flags |
|--------------------------|----------|-----------------|
| `openmp`                 | cpu      | no              |
| `mpi-calc-pi`            | cpu      | no              |
| `numpy`                  | cpu      | no              |
| `mpi-p2p`                | mpi      | no              |
| `gpu-burn-r8`            | gpu      | yes             |
| `nccl-tests`             | gpu      | yes             |
| `nvidia-hpc-benchmarks`  | gpu      | yes             |
| `gpu-fryer`              | gpu      | yes             |
| `megatron-lm`            | gpu      | yes             |

## Requirements

- Python 3.6+ (no third-party packages required)
- Access to a Slurm cluster (`sbatch`, `sinfo`, `sacctmgr` on `PATH`)
- Benchmark source trees rooted at `/orcd/data/orcd/022/benchmarks/<name>`
  (set via `ROOT_DIR` in `bench_config.py`)

## 1. Submitting jobs

```
python bench_submit.py <benchmark> [<benchmark> ...] [options]
```

### Arguments

| Flag             | Description                                         | Required |
|------------------|-----------------------------------------------------|----------|
| positional       | One or more benchmark names (see table above)       | yes \*\* |
| `--all-bench`    | Submit every registered benchmark                   | no \*\*  |
| `--nodes`        | Node numbers, e.g. `3511 3512` (no `node` prefix)   | no \*    |
| `--partition`    | Slurm partition                                     | yes      |
| `--reservation`  | Slurm reservation, or `none` (default: `none`)      | no       |
| `--qos`          | Slurm QoS (default: `normal`)                       | no       |
| `--cpus`         | CPUs per node                                       | no \*    |
| `--gpu-type`     | GPU type, e.g. `l40s`, `a100`, `h100`, `h200`       | GPU only |
| `--gpus`         | GPUs per node                                       | GPU only |

\* When omitted, the script queries Slurm (`sinfo`) for the value.
\*\* Provide either benchmark names or `--all-bench`, not both.

### Examples

Submit an NCCL test across two L40S nodes:
```
python bench_submit.py nccl-tests \
    --nodes 3511 3512 \
    --partition mit_normal_gpu \
    --qos unlimited \
    --cpus 48 \
    --gpu-type l40s --gpus 4
```

Submit both CPU benchmarks on a single node, auto-detecting CPU count:
```
python bench_submit.py openmp mpi-calc-pi \
    --nodes 3511 \
    --partition mit_normal
```

Submit a GPU burn across two H200 nodes:
```
python bench_submit.py gpu-burn-r8 \
    --nodes 3511 3512 \
    --partition mit_normal_gpu \
    --qos unlimited \
    --cpus 48 \
    --gpu-type h200 --gpus 8
```

Submit every node in a partition (auto-discovered):
```
python bench_submit.py gpu-fryer \
    --partition mit_normal_gpu \
    --gpu-type l40s --gpus 4
```

### Running **all** benchmarks at once

Pass `--all-bench` in place of benchmark names to submit every registered
benchmark. GPU benchmarks are dropped automatically when GPU info is
unavailable, so the same command works on CPU-only partitions:

```
# GPU partition — runs CPU + MPI + GPU benchmarks
python bench_submit.py --all-bench \
    --partition mit_normal_gpu --qos unlimited \
    --gpu-type l40s --gpus 4

# CPU-only partition — GPU benchmarks are skipped
python bench_submit.py --all-bench --partition mit_normal
```

## 2. Analyzing results

```
python bench_analyze.py <benchmark> [<benchmark> ...] --partition <name> [options]
```

### Arguments

| Flag             | Description                                         | Required |
|------------------|-----------------------------------------------------|----------|
| positional       | One or more benchmark names                         | yes \*\* |
| `--all-bench`    | Analyze every registered benchmark                  | no \*\*  |
| `--partition`    | Slurm partition whose output to parse               | yes      |
| `--num-results`  | Number of most-recent files to inspect (default: 2) | no       |
| `--gpu-type`     | GPU type — needed for `gpu-burn-r8` and `nvidia-hpc-benchmarks` | conditional |

\*\* Provide either benchmark names or `--all-bench`, not both.

### Examples

```
python bench_analyze.py nccl-tests --partition mit_normal_gpu --num-results 2

python bench_analyze.py openmp mpi-calc-pi --partition mit_normal --num-results 4

python bench_analyze.py gpu-burn-r8 \
    --partition mit_normal_gpu --num-results 3 --gpu-type l40s

python bench_analyze.py nvidia-hpc-benchmarks \
    --partition mit_normal_gpu --num-results 2 --gpu-type h200
```

### Analyzing **all** benchmarks at once

```
python bench_analyze.py --all-bench \
    --partition mit_normal_gpu --num-results 2 --gpu-type l40s

# CPU-only — skips GPU-type-dependent analyses automatically
python bench_analyze.py --all-bench --partition mit_normal --num-results 4
```

## Output locations

Each benchmark writes its Slurm output to a subdirectory under its own source
tree (`$ROOT_DIR/<benchmark>/...`), mirroring the layout of the original shell
scripts. The exact path is printed at submission time and read back by
`bench_analyze.py`.
