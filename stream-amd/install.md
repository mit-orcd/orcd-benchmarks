# stream-amd — installation

This is STREAM (McCalpin), built from the source in `STREAM/` with gcc and
OpenMP. The array size is fixed at compile time, so there is one binary per
size:

- `stream_c-100M-100N`: 100,000,000 elements per array (~2.4 GB total), 100 iterations
- `stream_c-430M-100N`: 430,080,000 elements (~10.3 GB). Each array is 3.4 GB,
  which meets STREAM's rule (each array ≥ 4 × total last-level cache) for nodes
  with up to ~860 MB of L3 in total.

Sources:

- <https://www.cs.virginia.edu/stream/>
- <https://github.com/jeffhammond/STREAM>
- AMD guidance: <https://www.amd.com/en/developer/zen-software-studio/applications/spack/stream-benchmark.html>

## Build

`make` builds the C and Fortran binaries for the size set in the `Makefile`.
Variables given on the command line override it, so both sizes can be built
without editing:

```bash
cd /orcd/data/orcd/022/benchmarks/stream-amd/STREAM
module load gcc/12.2.0
make clean SUB_BIN=100M-100N
make OPT="-DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=100" SUB_BIN=100M-100N
make OPT="-DSTREAM_ARRAY_SIZE=430080000 -DNTIMES=100" SUB_BIN=430M-100N
ls stream_c-* stream_f-*
```

Flags are `-O2 -fopenmp`. To get the source fresh:
`git clone https://github.com/jeffhammond/STREAM.git` and copy this
`Makefile` into it.

## Fix the binary path in the job scripts

The `job-*.sh` scripts that use local builds (all except `job-pi-mick.sh` and `job-bcs.sh`, which call the spack `stream_c.exe`) set `STREAM_PATH=/orcd/data/orcd/001/benchmarks/memory-bw/stream-amd/STREAM`,
an old location that no longer exists. Point them at this dir:

```bash
cd /orcd/data/orcd/022/benchmarks/stream-amd
sed -i 's#/orcd/data/orcd/001/benchmarks/memory-bw/stream-amd/STREAM#/orcd/data/orcd/022/benchmarks/stream-amd/STREAM#' job-*.sh
mkdir -p output
```

## Alternative: spack

```bash
spack install stream +openmp
# the site build is at /orcd/software/community/001/rocky8/stream/5.10/2yg/bin/stream_c.exe (default array size)
```
