# mpi-io — installation

The source is a single C file, `mpi-io-bw.c`, in this dir. Nothing is
downloaded.

## Build

`job.sh` runs `module load openmpi/5.0.6`. That module no longer exists,
although the install is still at `/orcd/software/community/001/pkg/openmpi/5.0.6`.
Rebuild with a current Open MPI and change the `module load` line in `job.sh`
to match:

```bash
cd /orcd/data/orcd/022/benchmarks/mpi-io
module load openmpi/5.0.8
mpicc -O3 mpi-io-bw.c -o mpi-io-bw
sed -i 's#^module load openmpi/5.0.6#module load openmpi/5.0.8#' job.sh
```

To keep the existing 5.0.6 binary instead, put the old install on `PATH` in
`job.sh`, in place of the `module load` line:

```bash
export PATH=/orcd/software/community/001/pkg/openmpi/5.0.6/bin:$PATH
export LD_LIBRARY_PATH=/orcd/software/community/001/pkg/openmpi/5.0.6/lib:$LD_LIBRARY_PATH
```

### CentOS-7 variant

`inp-mpi-io-bw.c` takes the output path and the MB per rank as arguments
(`./mpi-io-bw-c7 <file> [MB]`). It was built on the CentOS-7 nodes:

```bash
module load gcc/6.2.0 openmpi/3.0.4        # CentOS-7 module tree only
mpicc -O3 inp-mpi-io-bw.c -o mpi-io-bw-c7
```

## Test directory

`job.sh` copies the binary to `DIR=/orcd/scratch/orcd/022/shaohao/mpi-io`, the
filesystem under test, and runs there. Set `DIR` to a dir of your own on the
filesystem you want to test. The jobs write to `out/`:

```bash
mkdir -p /orcd/data/orcd/022/benchmarks/mpi-io/out
```
