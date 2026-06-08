# stream-amd

## Introduction

The [STREAM](https://www.cs.virginia.edu/stream/) memory-bandwidth
benchmark, tuned for AMD (EPYC) CPUs. STREAM measures sustainable main
memory bandwidth (MB/s) with four vector kernels — Copy, Scale, Add,
Triad. This setup runs prebuilt OpenMP binaries while sweeping thread
counts and array sizes to find peak per-node memory bandwidth.

## Installation

AMD's STREAM build guidance:
<https://www.amd.com/en/developer/zen-software-studio/applications/spack/stream-benchmark.html>.
Build via spack or from source (<https://github.com/jeffhammond/STREAM>):

```bash
spack install stream +openmp
```

The job scripts use prebuilt binaries (`stream_c-100M-100N`,
`stream_c-430M-100N`) from a shared STREAM install path; the source/build
also lives in `STREAM/`. Per the STREAM rule, each array must be ≥ 4× the
total last-level cache, hence the large 100M/430M element sizes.

## Usage

There is no `run/` many-runs driver; the root `job-*.sh` scripts are
per-target single-node runs that internally sweep `OMP_NUM_THREADS` and
array sizes.

### Single run — root `job-*.sh`

```bash
sbatch job-mit-normal-96.sh       # 96-core mit_normal node
# other presets: job-mit-normal-192.sh, job-high-l3.sh,
#                job-pi-srmadden.sh, job-pi-mick.sh, job-bcs.sh
```

Each loops `OMP_NUM_THREADS` (e.g. 48/96/192) and runs the STREAM
binaries for each array size. Output lands in `output/`. Tune memory
binding with `OMP_PLACES` / `OMP_PROC_BIND` (see comments in the scripts).

## Analysis

STREAM prints a best-rate table; the **Triad** MB/s line is the headline
memory-bandwidth number:

```bash
grep -A4 "Function" output/out.*        # Copy / Scale / Add / Triad MB/s
```

Compare Triad across thread counts and array sizes; the peak Triad MB/s is
the node's sustained memory bandwidth.
