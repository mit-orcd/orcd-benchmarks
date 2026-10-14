# stream-intel — installation

This is Intel's packaging of STREAM: `stream.c`, a `makefile` that builds one
binary per ISA, and the `run.sh` driver. It is built with the Intel compiler.
Nothing is downloaded. Upstream STREAM: <https://www.cs.virginia.edu/stream/>.

## Compiler

The `makefile` uses `CC = icc`. **`icc` is no longer installed**: oneAPI
removed it in 2024, and the cluster has only `intel/2024.2.1` and
`intel-hpc/2025.2.1.44`, which ship `icx`. Build with `icx` instead:

```bash
cd /orcd/data/orcd/022/benchmarks/stream-intel
module load intel/2024.2.1
make clean
make CC=icx                     # -> stream_avx.bin stream_avx2.bin stream_avx512.bin
# options: cpu=avx|avx2|avx512 (build only one), size=<elements per array>, rfo=1 (no NT stores)
```

The default is FP64, `STREAM_ARRAY_SIZE=269000000` (~2 GB per array, 6 GB in
total), `NTIMES=100`, `-O3 -qopenmp -mcmodel=medium`, and
`-qopt-streaming-stores always`.

**Caveat:** `icx` ignores `-qopt-streaming-stores` and does not emit
non-temporal stores (`vmovntpd`); this was checked with icx 2024.2. `run.sh`
checks the binary, so it reports `NT-Stores : ...` and writes its results under
`rfo/` instead of `nt/`. Copy, Scale and Triad will then read lower than with
the classic `icc` build, because of the extra read-for-ownership traffic.
For true NT-store numbers, build on a system that still has classic `icc`
(oneAPI ≤ 2023.2), or use the gcc build in `stream-amd`.

Build on (or for) the CPU you will test. `run.sh` picks the highest ISA binary
the CPU supports.

## Check

```bash
objdump -D stream_avx512.bin | grep -c vmovntpd     # > 0 means NT stores are present
```
