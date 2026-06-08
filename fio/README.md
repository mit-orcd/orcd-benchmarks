# fio

## Introduction

Filesystem I/O benchmark built on [fio](https://fio.readthedocs.io). It
measures sustained read/write bandwidth and IOPS of the cluster storage
targets (scratch / data filesystems) using direct, non-cached I/O across
multiple parallel jobs. Used to compare per-filesystem performance from
single and multiple nodes.

## Installation

`fio` is provided by the system (or `module load`/system package); the
fio source is also vendored in `src/`. No build step is normally needed —
just ensure `fio` is on `PATH`.

The job scripts write to a per-job directory on the filesystem under test,
e.g. `/orcd/<class>/orcd/<NN>/shaohao/fio/data-<jobid>`, and delete it
afterward.

## Usage

The core fio command (in every job script) sweeps a directory with
`--rw=readwrite --direct=1 --bs=1M --iodepth=64 --size=10G` over
`numjobs = ntasks`. Pick the filesystem with two args: `$1` = class
(e.g. `scratch`/`data`), `$2` = storage number (e.g. `011`).

### Automated, many runs — `loop-*.sh`

`loop-1node.sh` submits `job-1node.sh` once per storage target;
`loop-mnodes-perfs.sh` drives the multi-node array job (`job-mnode.sh`).
Edit the partition flags and the target list, then:

```bash
./loop-1node.sh             # one node per target -> output-1node/
./loop-mnodes-perfs.sh      # multi-node array     -> output-mnode/
```

### Single run — root dir

```bash
# job.sh / job-1node.sh:  args = <class> <storage-number>
sbatch -p mit_normal job-1node.sh scratch 011
# submit-job.sh holds ready-made sbatch one-liners for specific nodes
```

`fstor.sh` / `hstor.sh` are quick standalone variants for specific
storage. `c7-job.sh` is the CentOS-7 variant.

## Analysis

```bash
./get-result.sh             # buckets IOPS by node/partition into results/
# or quick look:
grep -e "directory" -e "IOPS" output-1node/*
```

Each fio run prints aggregate read/write **bw** (MB/s) and **IOPS** in its
`group_reporting` summary; `get-result.sh` greps the `directory` and
`IOPS` lines and splits them per partition/node under `results/`.
