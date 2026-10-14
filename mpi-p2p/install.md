# mpi-p2p — installation

The benchmark is the OSU Micro-Benchmarks (`osu_bw`, `osu_latency`, …),
compiled once for each MPI stack you want to test. The builds go to
`install/<name>/` in this dir.

- Source: <https://mvapich.cse.ohio-state.edu/benchmarks/>
- Tarballs, already in `src/`: `osu-micro-benchmarks-7.5-1.tar.gz` (CPU builds)
  and `osu-micro-benchmarks-7.5.2.tar.gz` (CUDA-aware nvhpc builds)

## 1. Get the source (only if `src/` has no tarball)

```bash
cd /orcd/data/orcd/022/benchmarks/mpi-p2p/src
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.5-1.tar.gz
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.5.2.tar.gz
```

## 2. Build for an Open MPI or Intel MPI module — `build.sh`

`build.sh <os> <module> <version>` unpacks the 7.5-1 tarball, runs
`module load <module>/<version>`, then `configure CC=mpicc CXX=mpicxx`,
`make` and `make install` into `install/<os>-<version>` (Intel:
`install/<os>-intel-<version>`). It **deletes an existing install of the same
name first**.

```bash
cd /orcd/data/orcd/022/benchmarks/mpi-p2p/src
module load gcc/12.2.0                        # openmpi/* modules need it loaded first
./build.sh r8 openmpi 4.1.4                   # -> install/r8-4.1.4   (used by run/run.sh)
./build.sh r8 intel-hpc 2025.2.1.44           # -> install/r8-intel-2025.2.1.44 (mpiicx/mpiicpx)
```

Build on a compute node, not the login node. Either use `salloc`, or edit the
command in `job-build.sh` and `sbatch job-build.sh`. The configure, make and
install logs go to `src/<name>/log.{config,make,install}`.

## 3. Build for a module picked by its module path — `build-ompi-stack.sh`

Two different modules are both called `openmpi/5.0.8`. This script picks one by
its module path, installs to `install/<name>`, and **refuses to overwrite an
existing build**:

```bash
./build-ompi-stack.sh r8-spack-ompi-5.0.8 /orcd/software/core/001/spack/modulefiles/gcc/12.2.0 openmpi/5.0.8
./build-ompi-stack.sh r8-core-ompi-5.0.8  /orcd/software/core/001/modulefiles                 openmpi/5.0.8
# batch version: sbatch job-build-ompi-stacks.sh
```

The core `openmpi/5.0.8` needs `/opt/mellanox/hcoll`, which only some nodes
have (e.g. `mit_normal`), so build and run it there.

## 4. CUDA-aware build with NVHPC (HPC-X Open MPI) — `build-nvhpc.sh`

```bash
sbatch --gres=gpu:1 job-build.sh          # runs ./build-nvhpc.sh r8 nvhpc 26.1
# or by hand on a GPU node:
./build-nvhpc.sh r8 nvhpc 26.1            # -> install/r8-nvhpc-26.1, configure --enable-cuda --with-cuda=$NVHPC_ROOT/cuda
```

`build-nvhpc-ompi.sh` does the same for an nvhpc compiler plus a separate
Open MPI module. `build-c7.sh` is for the old CentOS-7 nodes.

## 5. Check

```bash
source /orcd/data/orcd/022/benchmarks/mpi-p2p/run/env.sh r8 4.1.4   # loads gcc/12.2.0 + openmpi/4.1.4, adds OSU to PATH
which osu_bw osu_latency
```

`env.sh <os> <version>` expects `install/<os>-<version>/libexec/osu-micro-benchmarks/mpi/`.
Do not use `install/r8-5.0.8`: it is linked against nvhpc's openmpi-3.1.5.
