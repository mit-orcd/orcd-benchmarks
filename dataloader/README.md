# dataloader

## Introduction

A PyTorch `DataLoader` I/O benchmark. It loads a tokenized dataset
(LLaMA-style `.pt` tensors, or ImageNet) through a `DataLoader` while
varying `num_workers`, and reports the achieved read throughput in GB/s.
It shows how dataset-streaming bandwidth scales with the number of worker
processes for a given filesystem.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/dataloader
sbatch job-llama.sh /orcd/scratch/orcd/011/$USER/wikipedia_tokenized.pt
# when done:
grep num_workers out-llama/bw-llama-<node>-<jobid>     # num_workers=<n> --> <x> GB/s ...
```

## Installation

See [install.md](install.md): the `torch` conda env, plus `wikipedia_tokenized.pt`
and an ImageNet copy on each filesystem under test.

## Usage

| Script | Reads | `num_workers` swept | Batch |
|---|---|---|---|
| `bw-llama.py <file.pt>` | one `torch.save`d list of 2048-token tensors | 0, 2, 4 | 64 |
| `bw-imagenet.py <ilsvrc_root>` | `<root>/train/` JPEGs via `ImageFolder` (resize + crop 224) | 96, 64, 32, 16, 8, 4, 2, 0 | 128 |

Both run on CPU only. The job scripts load `miniforge/24.3.0-0` and activate
`torch`.

### Automated, many runs — `loop-llama.sh`

`loop-llama.sh` submits `job-llama.sh` once per scratch filesystem
(`/orcd/scratch/orcd/{002,004,...,013}/shaohao/wikipedia_tokenized.pt`). Edit
the list and the user name, then:

```bash
./loop-llama.sh
```

### Single run — root dir

```bash
sbatch job-llama.sh <path>/wikipedia_tokenized.pt    # mit_normal, 8 cores, 2 h -> out-llama/
sbatch job-imagenet.sh                               # set dir= inside; exclusive node, 64 cores, 6 h -> out-imagenet/
```

`c7-job-imagenet.sh` is an older CentOS-7 variant of `job-imagenet.sh`.

## Analysis

`bw-llama.py` prints:

```
num_workers=0 --> 7.425 GB/s over 0.36 GB in 0.05 s
```

`bw-imagenet.py` prints:

```
[Split=train, num_workers=64] Time: 130.89s | Total: 718.428 GB | Speed: 5.489 GB/s
```

GB is the size of the decoded tensors, not the bytes read from disk. To print
the last 10 runs:

```bash
./get-results.sh                       # set mode=llama|imagenet at the top
grep -H num_workers out-llama/* | grep GB
```

How to read the results:

- **ImageNet** measures real file streaming: the time includes opening and
  decoding each JPEG. Compare GB/s across `num_workers` to find where adding
  workers stops helping. That plateau is the filesystem or metadata limit.
  Compare filesystems at the same worker count.
- **LLaMA**: `torch.load` reads the whole `.pt` into RAM before the timed loop,
  so the GB/s is in-memory DataLoader throughput. It usually drops as workers
  are added, because of the process start-up and IPC cost. The file read time
  is not reported. Use `sacct -j <jobid> -o Elapsed` or time the `Loading ...`
  step if you need it.
