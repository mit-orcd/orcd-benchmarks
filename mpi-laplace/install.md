# mpi-laplace — installation

The source is in `src/` (C and Fortran, from an MPI teaching set). Nothing is
downloaded. The benchmark uses only `laplace_serial.c` and `laplace_mpi.c`:
a 1000 × 1000 Jacobi solve. The MPI version is hard-coded to 4 ranks
(`#define NPROC 4`).

## Build

The prebuilt binaries are in `src/bin-r8/` (Open MPI 4.1.4) and
`src/bin-r8/mpi-5.0.6/` (Open MPI 5.0.6, a module that no longer exists).
To rebuild against the current Open MPI 4.1.4:

```bash
cd /orcd/data/orcd/022/benchmarks/mpi-laplace/src/bin-r8
module load gcc/12.2.0 openmpi/4.1.4
gcc   -O3 ../laplace_serial.c -o laplace_serial -lm
mpicc -O3 ../laplace_mpi.c    -o laplace_mpi    -lm
```

The `notes` file uses the old Rocky-8 module names (`gcc/12.2.0-x86_64`,
`openmpi/4.1.4-pmi-ucx-x86_64`). On the current tree those need
`module load deprecated-modules` first. `work/job_all_example.sh` loads
`StdEnv gcc openmpi/4.1.4`, which matches the build above.

To change the rank count, edit `#define NPROC` in `laplace_mpi.c` (it must
divide 1000) and rebuild.

## Input and output

`work/inp` holds the maximum iteration count (`4000`), read on stdin. The
driver writes to `work/<partition>/output/`:

```bash
mkdir -p /orcd/data/orcd/022/benchmarks/mpi-laplace/work/<partition>/output
```

`job_all_example.sh` has a bug: it runs `mkdir -p output_dir`, without `$`, so
create the dir by hand.
