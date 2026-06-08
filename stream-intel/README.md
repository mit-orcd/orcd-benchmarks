# stream-intel

## Introduction

The [STREAM](https://www.cs.virginia.edu/stream/) memory-bandwidth
benchmark, packaged by Intel with compile/run scripts tuned for Intel
CPUs. STREAM measures sustainable main memory bandwidth (MB/s) via four
vector kernels — Copy, Scale, Add, Triad — built with the Intel compiler
so it emits non-temporal stores for peak bandwidth.

## Installation

Requires the Intel C compiler (`icc`) on `PATH`. Build with `make`, which
produces one binary per ISA:

```bash
module load intel        # provides icc
make                     # -> stream_avx.bin, stream_avx2.bin, stream_avx512.bin
# options: make cpu=avx512   make size=<elems_per_array>   make rfo=1
```

Default config: FP64, `STREAM_ARRAY_SIZE=269000000` (~2 GB/array),
`NTIMES=100`. Source is `stream.c` (vendored from the STREAM project).

## Usage

There is no `run/` many-runs driver; `run.sh` (repo root) is the single
benchmarking entry point.

### Single run — root `run.sh`

```bash
./run.sh
```

`run.sh` auto-detects the machine (sockets/cores/caches/NUMA), picks the
highest-ISA binary your CPU supports, sets `OMP_NUM_THREADS` to the
physical core count with compact `KMP_AFFINITY` (ignoring hyperthreads),
runs STREAM, and writes a log with the result plus system info. Running
under `sudo` adds `dmidecode` memory details.

## Analysis

The run log contains STREAM's best-rate table; the **Triad** MB/s value is
the headline sustained-memory-bandwidth number (Copy/Scale/Add are also
reported). Higher is better; compare against the platform's theoretical
DRAM bandwidth to gauge efficiency.
