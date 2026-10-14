# stream-intel

## Introduction

The [STREAM](https://www.cs.virginia.edu/stream/) memory-bandwidth benchmark, as
packaged by Intel for Intel CPUs. STREAM measures sustainable main-memory
bandwidth (MB/s) with four vector kernels: Copy, Scale, Add and Triad. Intel's
`run.sh` detects the CPU (sockets, cores, caches, NUMA, HT), picks the
highest-ISA binary, pins one OpenMP thread per physical core, and logs the
result together with the system info.

## Quick Start

```bash
cd /orcd/data/orcd/022/benchmarks/stream-intel
srun -p mit_normal -N 1 --exclusive -t 20 ./run.sh       # Intel node, whole node
# when done:
grep -E '^(Copy|Scale|Add|Triad):' */*/*_*t.log            # Triad = headline MB/s
```

## Installation

See [install.md](install.md): `module load intel/2024.2.1; make CC=icx`.
Classic `icc` is no longer available, and `icx` builds lack non-temporal
stores (see the caveat there).

## Usage

There is no `run/` many-runs driver. The root `run.sh` is the single entry
point.

### Single run — root `run.sh`

```bash
./run.sh                 # on a compute node of the type to test; run as root (sudo) to add dmidecode memory info
```

`run.sh` runs `bench_simple`: one run with `OMP_NUM_THREADS` = all physical
cores and `KMP_AFFINITY=compact` (skipping hyper-threads). The log goes to

```
<cpu-model>/<nt|rfo>/stream_<isa>_<threads>t.log
```

in the current dir. `<cpu-model>` is the `lscpu` model name with spaces turned into `-`.
`nt` or `rfo` depends on whether the binary has non-temporal stores. To sweep
1…N threads on one socket (compact pinning) and each NUMA domain
separately, change the last line of `run.sh` from `bench_simple` to
`bench_sweep`. That writes under `<cpu-model>/nps-<N>/{uma,numa}/...`.

To submit it as a batch job on a chosen node:

```bash
sbatch -p mit_normal -N 1 --exclusive -t 30 -w node1234 --wrap "cd $PWD && ./run.sh"
```

## Analysis

Each log starts with the system info (CPU, caches, NUMA, NT-store status) and
ends with STREAM's best-rate table:

```
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:  ...
Triad: ...
Solution Validates: avg error less than 1.000000e-13 on all three arrays
```

```bash
grep -H -E '^Triad:' */*/*.log */*/*/*/*/*.log 2>/dev/null
```

**Triad** MB/s is the headline sustained bandwidth. Compare it with the
theoretical peak, sockets × channels × MT/s × 8 B: e.g. 2 × 8 × DDR5-4800 ≈
614 GB/s. Well-tuned NT-store runs reach about 80–90 % of peak; RFO (`icx`)
runs read lower on Copy, Scale and Triad. Compare only runs of the same store
type. Ignore runs that do not print `Solution Validates`.
