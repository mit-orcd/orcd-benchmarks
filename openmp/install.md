# openmp — installation

The source is in `src/` and nothing is downloaded. The benchmark uses
`src/pi_omp.c`, a midpoint-rule integral of 4/(1+x²) with an OpenMP `reduction`.
`mat_mul.c`, `saxpy_omp.c` and the Fortran versions are extra examples.

## Build

```bash
cd /orcd/data/orcd/022/benchmarks/openmp/src
module load gcc/12.2.0
gcc -O3 -fopenmp pi_omp.c -o pi_omp
```

`run/run.sh` and `work/job*.sh` run `src/pi_omp` by its absolute path. The
existing binary links `libgomp` from the old Rocky-8 spack gcc tree
(`/nfs/software001/...`). Rebuilding with the current `gcc/12.2.0` removes that
dependency.

Optional extras:

```bash
gcc -O3 -fopenmp mat_mul.c   -o mat_mul
gcc -O3 -fopenmp saxpy_omp.c -o saxpy_omp
gfortran -O3 -fopenmp pi_omp.f90 -o pi_omp_f
```

## Check

```bash
OMP_NUM_THREADS=4 ./pi_omp        # pi = 3.14159265358..., time to compute = ... seconds
```
