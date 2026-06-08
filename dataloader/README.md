# dataloader

## Introduction

A PyTorch `DataLoader` I/O benchmark. It loads a tokenized dataset
(LLaMA-style `.pt` tensors, or ImageNet) through a `DataLoader` while
varying `num_workers`, and reports the achieved read throughput in GB/s.
It shows how dataset-streaming bandwidth scales with the number of worker
processes for a given filesystem.

## Installation

Needs PyTorch in a conda/torch environment:

```bash
module load miniforge/24.3.0-0
source activate torch          # env with torch installed
```

The benchmarks are the Python scripts in this directory: `bw-llama.py`
(takes a `.pt` file path) and `bw-imagenet.py`. Datasets live on scratch,
e.g. `/orcd/scratch/orcd/<NN>/shaohao/wikipedia_tokenized.pt`.

## Usage

Both scripts time a full pass over the dataset for a list of
`num_workers` values (0, 2, 4, …) and print GB/s for each.

### Automated, many runs — `loop-llama.sh`

Submits `job-llama.sh` once per storage target (sweeping the same dataset
on each filesystem):

```bash
./loop-llama.sh             # loops over /orcd/scratch/orcd/<NN>/...
```

Output lands in `out-llama/` (and `out-imagenet/` for the ImageNet path).

### Single run — root dir

```bash
# job-llama.sh: arg = path to the .pt dataset
sbatch job-llama.sh /orcd/scratch/orcd/011/shaohao/wikipedia_tokenized.pt
# job-imagenet.sh / c7-job-imagenet.sh: ImageNet variants
```

## Analysis

```bash
./get-results.sh            # set mode=llama|imagenet at top; greps GB/s lines
# or:
grep num_workers out-llama/* | grep GB
```

Each line reads `num_workers=<n> --> <x> GB/s …`; higher GB/s is better.
Compare across `num_workers` to find where worker parallelism stops
helping (filesystem-bound).
