# all-bench — installation

`all-bench` has nothing to compile. It needs:

1. The target benchmarks, each built or pulled in its own dir under
   `/orcd/data/orcd/022/benchmarks`. Follow that benchmark's `install.md`:
   `openmp`, `mpi-calc-pi`, `mpi-p2p`, `gpu-burn-r8`, `nvidia-hpc-benchmarks`,
   `nccl-tests`, `megatron-lm`, `b200-nodes`, `b200-kimi`.
2. Optional: the `orcd-benchmark` module, which puts `run-all.sh` and
   `get-results-*.sh` on `PATH` and sets `$BENCH_ROOT`.

## Optional: install the module

The modulefile only sets environment variables. It points at the scripts in
place and copies none of them.

```bash
# The deployed name is orcd-benchmark. It differs from 1.0.lua only in help text.
cp /orcd/data/orcd/022/benchmarks/all-bench/modulefiles/all-bench/1.0.lua \
   /orcd/compute/orcd/011/team-software/modulefiles/orcd-benchmark.lua
chmod 664 /orcd/compute/orcd/011/team-software/modulefiles/orcd-benchmark.lua

# Check it
module use /orcd/compute/orcd/011/team-software/modulefiles
module load orcd-benchmark
which run-all.sh            # -> /orcd/data/orcd/022/benchmarks/all-bench/run-all.sh
echo $BENCH_ROOT            # -> /orcd/data/orcd/022/benchmarks
```

If the benchmarks move, edit `local root = "..."` in the modulefile.

To use the module without installing it:

```bash
module use /orcd/data/orcd/022/benchmarks/all-bench/modulefiles
module load all-bench
```

The scripts are readable by group `sched_mit_orcd` only, so the module works
only for members of that group.
