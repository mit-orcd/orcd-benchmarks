# fio

## Introduction

Filesystem I/O benchmark built on [fio](https://fio.readthedocs.io). It
measures sustained read/write bandwidth and IOPS of the cluster storage
targets (scratch / data filesystems) using direct, non-cached I/O across
multiple parallel jobs. Used to compare per-filesystem performance from
single and multiple nodes.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/fio
sbatch -p mit_normal --exclusive job-1node.sh scratch 011     # test /orcd/scratch/orcd/011
# when done (~2 min):
grep -E 'Data directory|READ: bw|WRITE: bw' output-1node/scratch-<node>-<jobid>
```

## Installation

See [install.md](install.md). The system `fio` 3.19 is used, so normally only the
per-user test dirs need to be created.

## Usage

Every job script runs the same command against a fresh per-job directory, then
deletes it:

```bash
fio --name=data --rw=readwrite --numjobs=$SLURM_NTASKS --direct=1 --directory=$DIR \
    --size=10G --time_based --runtime=60s --ramp_time=2s --verify=0 \
    --bs=1M --iodepth=64 --group_reporting=1 --ioengine=psync
```

`$1` = class (`scratch` or `data`) and `$2` = storage number (`004`, `011`, …)
select `DIR=/orcd/$1/orcd/$2/shaohao/fio/data-<jobid>`. The workload is 50/50
mixed read/write, 1 MiB blocks, O_DIRECT, one file per task, 60 s. The default
is 16 tasks on one node (`-n 16`).

### Automated, many runs — `loop-*.sh`

| Script | Submits | Output |
|---|---|---|
| `loop-1node.sh` | `job-1node.sh` once per storage number in its `for i in ...` list | `output-1node/<class>-<node>-<jobid>` |
| `loop-mnodes-perfs.sh` | `job-mnode.sh` as an exclusive array of `N_jobs` nodes on target `i` (`mit_normal`) and `j` (`mit_normal_gpu`) | `output-mnode/<class>-<node>-<arrayid>-<task>` |

Edit `flag`, `type` and the target numbers at the top, then:

```bash
./loop-1node.sh
./loop-mnodes-perfs.sh
```

The array tasks run at the same time, so the multi-node case measures the
aggregate load on one filesystem.

### Single run — root dir

```bash
sbatch -p mit_normal job-1node.sh scratch 011    # = job.sh, output to output-1node/
```

`submit-job.sh` holds ready-made `sbatch` lines for specific nodes.
`fstor.sh` and `hstor.sh` are interactive variants for two specific storage
paths. `c7-job.sh` is the CentOS-7 version.

## Analysis

The result is the `Run status group 0` block at the end of each output:

```
   READ: bw=11.1GiB/s (11.9GB/s), ..., io=665GiB (714GB), run=60001-60001msec
  WRITE: bw=11.1GiB/s (11.9GB/s), ..., io=664GiB (712GB), run=60001-60001msec
```

Read the aggregate bandwidth from these two lines. The per-direction `IOPS=`
lines higher up give IOPS; with 1 MiB blocks, IOPS ≈ MiB/s.

```bash
# one row per run: target, read BW, write BW
for f in output-1node/*; do
  printf "%-45s %s  %s\n" "$(grep -m1 'Data directory' $f | awk '{print $3}')" \
    "$(grep -m1 ' READ: bw' $f | awk '{print $2}')" "$(grep -m1 'WRITE: bw' $f | awk '{print $2}')"
done
```

`get-result.sh` collects the `directory` and `IOPS` lines of all runs into
`results/bw.1node` and `results/bw.mnode`, then splits them per partition by the
node lists in `results/nodes.mit_normal` and `results/nodes.mit_normal_gpu`.
For multi-node runs, add up the BW of all array tasks with the same `<arrayid>`
to get the filesystem aggregate.
