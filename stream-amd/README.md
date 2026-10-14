# stream-amd

## Introduction

The [STREAM](https://www.cs.virginia.edu/stream/) memory-bandwidth
benchmark, tuned for AMD (EPYC) CPUs. STREAM measures sustainable main
memory bandwidth (MB/s) with four vector kernels — Copy, Scale, Add,
Triad. This setup runs prebuilt OpenMP binaries while sweeping thread
counts and array sizes to find peak per-node memory bandwidth.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/stream-amd
sbatch job-mit-normal-96.sh             # 96-core mit_normal node: 48/96/192 threads x 100M/430M arrays
# when done:
grep -E 'OMP_NUM_THREADS =|array size|^Triad' output/out.mit-normal-<node>-<jobid>
```

## Installation

See [install.md](install.md): `make` in `STREAM/` builds `stream_c-100M-100N`
and `stream_c-430M-100N`. Then fix `STREAM_PATH` in the `job-*.sh` scripts,
which still point to an old location.

## Usage

There is no `run/` many-runs driver; the root `job-*.sh` scripts are
per-target single-node runs that internally sweep `OMP_NUM_THREADS` and
array sizes.

### Single run — root `job-*.sh`

| Script | Partition | `-n` | Thread sweep |
|---|---|---|---|
| `job-mit-normal-96.sh` | mit_normal | 96 | 48 96 192 |
| `job-mit-normal-192.sh` | mit_normal | 192 | 96 192 384 |
| `job-high-l3.sh` | mit_normal | 64 | 32 64 128 |
| `job-pi-srmadden.sh` | pi_srmadden | 64 | 32 64 128 |
| `job-pi-mick.sh` | pi_mick | 64 | one run, all cores, spack `stream_c.exe` |
| `job-bcs.sh` | ou_bcs_low | 48 | one run, all cores, spack `stream_c.exe` |

```bash
sbatch job-mit-normal-96.sh
sbatch -w node1600 job-mit-normal-96.sh      # a specific node
```

Each script prints `lscpu`, then for every thread count runs both array sizes
(`--mem=0`, whole-node memory). The output goes to
`output/out.<job-name>-<node>-<jobid>`. Thread pinning (`OMP_PLACES`,
`OMP_PROC_BIND`) is left unset on purpose. The comments in the scripts note
that the tried bindings were slower.

## Analysis

Each STREAM run ends with a table of the best rate over 100 iterations:

```
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:          360780.9     0.004946     0.004435     0.009350
Scale:         329142.5     0.005604     0.004861     0.015147
Add:           375063.5     0.006994     0.006399     0.012885
Triad:         360464.4     0.007428     0.006658     0.014760
```

followed by `Solution Validates`.

```bash
# Triad MB/s for every (threads, array size) in one file
awk '/OMP_NUM_THREADS =/{t=$3} /array size/{a=$4} /^Triad:/{print t, a, $2}' output/<file>
```

**Triad** is the headline number. The node's memory bandwidth is the best Triad
over the thread counts at the large (430M) size. It usually peaks at one thread
per core or per CCD, not with SMT. Compare against the platform's theoretical
peak, channels × MT/s × 8 B: e.g. 12-channel DDR5-4800 per socket is about
460 GB/s per socket. A node well below its peers points to missing or
mis-populated DIMMs or a NUMA/BIOS setting. Ignore runs that do not print
`Solution Validates`.
