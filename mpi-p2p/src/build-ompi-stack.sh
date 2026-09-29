#!/bin/bash
# Build OSU micro-benchmarks against one specific OpenMPI module, picked by
# its module path (needed because two different modules are both called
# openmpi/5.0.8). Never touches an existing build: refuses to run if the
# source or install directory already exists (unlike build.sh, which rm -rf's).
#
# build-ompi-stack.sh <name> <modulepath> <module>
#   e.g. build-ompi-stack.sh r8-spack-ompi-5.0.8 \
#            /orcd/software/core/001/spack/modulefiles/gcc/12.2.0 openmpi/5.0.8
# Installs into ../install/<name>. Run it on a compute node (see
# job-build-ompi-stacks.sh), not the login node.

set -e
name=$1
modulepath=$2
module=$3
if [[ -z "$name" || -z "$modulepath" || -z "$module" ]]; then
    echo "usage: $0 <name> <modulepath> <module>" >&2
    exit 2
fi

OSU_BENCH=osu-micro-benchmarks-7.5-1
SRC_ROOT=$(cd "$(dirname "$0")" && pwd)
SRC_DIR=$SRC_ROOT/$name
INSTALL_DIR=$(cd "$SRC_ROOT/../install" && pwd)/$name

for d in "$SRC_DIR" "$INSTALL_DIR"; do
    if [ -e "$d" ]; then
        echo "refusing: $d already exists (existing builds are never modified)" >&2
        exit 1
    fi
done

type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
module load gcc/12.2.0
module use "$modulepath"
module load "$module"
echo "mpicc: $(which mpicc)"
mpicc --showme:command

# A module whose libmpi can't resolve its own dependencies can't build (or
# run) anything -- say so instead of failing deep inside configure.
libmpi=$(mpicc --showme:libdirs | tr ' ' '\n' | while read d; do
             [ -e "$d/libmpi.so" ] && echo "$d/libmpi.so" && break; done)
missing=$(ldd "$libmpi" 2>/dev/null | awk '/not found/ {print $1}' | tr '\n' ' ')
if [ -n "$missing" ]; then
    echo "refusing: $libmpi needs libraries that are not installed: $missing" >&2
    exit 3
fi

# unpack into a private temp dir, then move into place under the new name,
# so an existing $OSU_BENCH directory in src/ is never touched either
tmp=$(mktemp -d "$SRC_ROOT/.unpack-$name-XXXX")
tar xzf "$SRC_ROOT/${OSU_BENCH}.tar.gz" -C "$tmp"
mv "$tmp/$OSU_BENCH" "$SRC_DIR"
rmdir "$tmp"

mkdir -p "$INSTALL_DIR"
cd "$SRC_DIR"
./configure CC=mpicc CXX=mpicxx --prefix="$INSTALL_DIR" > log.config 2>&1
make -j "${SLURM_CPUS_ON_NODE:-4}" > log.make 2>&1
make install > log.install 2>&1

bin=$INSTALL_DIR/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency
echo "built $bin"
readelf -d "$bin" | grep -E "RPATH|RUNPATH"
ldd "$bin" | grep libmpi
