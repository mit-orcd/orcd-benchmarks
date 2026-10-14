# numpy — installation

The only requirement is NumPy. The benchmark is `mat_mult.py` in this dir.

## Option A: the miniforge module (what the job scripts use)

The `miniforge/23.11.0-0` base env already includes NumPy 1.26.4, linked to
OpenBLAS 0.3.26 (`libopenblasp`, threaded with OpenMP). There is nothing to
install:

```bash
module load miniforge/23.11.0-0
python -c "import numpy; print(numpy.__version__); numpy.show_config()"
```

## Option B: a private conda env

This is what `mit_normal/mat_mult/create_numpy_env.sbatch` does:

```bash
module load miniforge/24.3.0-0
conda create -y -n benchmark numpy
source activate benchmark
```

If you use it, replace `module load miniforge/23.11.0-0` in the job scripts with
`module load miniforge/24.3.0-0; source activate benchmark`. Results from a
different NumPy/BLAS build are not directly comparable.

## Before the first run

`job_full_all_example.sh` and `job_half_all_example.sh` set
`output_dir=/orcd/data/orcd/001/benchmarks/numpy/$partition/output`, a path that
no longer exists. Change it to this dir:

```bash
cd /orcd/data/orcd/022/benchmarks/numpy
sed -i 's#/orcd/data/orcd/001/benchmarks/numpy#/orcd/data/orcd/022/benchmarks/numpy#' job_*_all_example.sh
```

One run of `mat_mult.py` needs about 10 GB of memory (three 20000 × 20000
float64 arrays).
