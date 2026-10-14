# fio — installation

The job scripts call `fio` from `PATH`. On the compute nodes that is the OS
package `/usr/bin/fio`, version 3.19 (`fio-3.19-4.el8`), so normally nothing
needs to be installed.

- Source: <https://github.com/axboe/fio>
- Docs: <https://fio.readthedocs.io/en/latest/fio_doc.html>

## Option A: OS package (what the results used)

```bash
fio --version                 # check on a compute node: fio-3.19
sudo dnf install -y fio       # admins only, if it is missing
```

## Option B: build from source

`src/` is a clone at `fio-3.37-28-g3ed8e`. Its `src/fio` binary is zero-filled (corrupt) and
does not run, so rebuild it:

```bash
cd /orcd/data/orcd/022/benchmarks/fio
git clone https://github.com/axboe/fio.git src        # skip if src/ exists
cd src
git checkout fio-3.37          # or keep the existing checkout
make clean
./configure --prefix=$PWD/install
make -j 8
make install
./install/bin/fio --version
export PATH=/orcd/data/orcd/022/benchmarks/fio/src/install/bin:$PATH   # add to the job scripts to use it
```

## Test directories and output dirs

The jobs write to `/orcd/<class>/orcd/<NN>/shaohao/fio/data-<jobid>` and
delete it at the end. The parent dir must exist and be writable on every
filesystem you test. Replace `shaohao` with your user name in `job.sh`,
`job-1node.sh` and `job-mnode.sh`.

```bash
for n in 004 009 010 011 012 013; do mkdir -p /orcd/scratch/orcd/$n/$USER/fio; done
cd /orcd/data/orcd/022/benchmarks/fio
mkdir -p output-1node output-mnode results
```

Each job needs about `ntasks × 10 GB` of free space on the target, e.g. 160 GB
for 16 tasks.
