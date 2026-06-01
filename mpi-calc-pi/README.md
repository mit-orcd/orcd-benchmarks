# mpi-calc-pi

MPI parallel estimation of π. Measures CPU scaling with process count.

## Build

```bash
module load openmpi/4.1.4
cd src
mpicc -O3 calc_pi_mpi.c -o calc_pi_mpi
```

## Run (single benchmark)

```bash
cd run
sbatch run.sh       # submits a Slurm job, varying NUM_THREADS=1..2*cores
```

## Analyze

```bash
cd run
./get-results.sh    # prints THREADS and elapsed time for recent outputs
```

## Run via py-all-bench

```bash
cd ../py-all-bench
module load miniforge/24.3.0-0
python bench_submit.py  mpi-calc-pi --partition mit_normal --nodes 3511
python bench_analyze.py mpi-calc-pi --partition mit_normal --num-results 4
```
