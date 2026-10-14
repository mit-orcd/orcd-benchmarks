# mpi-calc-pi — installation

The source is a single C file, `src/calc_pi_mpi.c`. It integrates
4/(1+x²) over n = 2×10¹¹ intervals, so there is nothing to download.

## Build

```bash
cd /orcd/data/orcd/022/benchmarks/mpi-calc-pi/src
module load gcc/12.2.0 openmpi/4.1.4
mpicc -O3 calc_pi_mpi.c -o calc_pi_mpi -lm
cp calc_pi_mpi calc_pi_mpi_big      # work/job*.sh run calc_pi_mpi_big; it is the same binary
```

- `run/run.sh` runs `module load openmpi/4.1.4`, which relies on `gcc/12.2.0`
  being in the job's default environment. Build and run with the same Open MPI.
- `calc_pi_mpi.c.bak` is the older 10× smaller problem (n = 2×10¹⁰).
- `src/env.sh` loads the old Rocky-8 Open MPI (`openmpi/4.1.4-pmi-ucx-x86_64`
  from `deprecated-modules`). Use it only to rebuild for the CentOS-7/Rocky-8
  node images.

## Check

```bash
srun -n 4 -p mit_normal -t 10 mpirun -np 4 ./calc_pi_mpi
# time taken: ...
# pi is approximately 3.14159265..., Error is ...
```

## Output dirs

`run/run.sh` creates `work/<partition>/output/` itself.
