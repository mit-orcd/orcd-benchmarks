#!/usr/bin/env bash
# Build rocHPL for MI355X (gfx950) with a GPU-aware MPI, into /orcd/data/orcd/022/benchmarks/rochpl.
# ROCm: amd-software/rocm-7.14.0 (ROCM_STACK=7.14), not the host /opt/rocm 7.2.4.
#   UCX 1.18.0      --with-rocm (GPU buffers) + verbs/rdmacm (ionic RoCE NICs)  -> $R/ucx
#   Open MPI 5.0.8  --with-ucx --with-rocm (pml ucx, rocm accelerator)         -> $R/openmpi
#   rocHPL (git)    ROCm 7.14 hipcc + rocBLAS                                  -> $R/rocHPL
# Run ON an AMD node, e.g. from a login node:
#   ssh node6100.inband 'setsid nohup bash /orcd/data/orcd/022/benchmarks/rochpl/setup/build-rochpl.sh \
#        > /orcd/data/orcd/022/benchmarks/rochpl/setup/logs/build.log 2>&1 < /dev/null &'
# Builds in node-local /scratch (NFS builds are ~20x slower); only the install prefixes are on /orcd/data.
# Each step is skipped if its install already exists (REBUILD=1 wipes them first). Writes $R/BUILD_OK.
set -euo pipefail
export ROCM_STACK=7.14
source /orcd/data/orcd/022/benchmarks/amd-software/env.sh
R=/orcd/data/orcd/022/benchmarks/rochpl
ROCM=$ROCM_PATH
export PATH=$ROCM/bin:$PATH
# our Open MPI/UCX first: the node env also puts amd-software/openmpi/lib (no UCX/ROCm) on the path;
# ROCm lib for libamdhip64 (Open MPI's rocm accelerator)
export LD_LIBRARY_PATH=$R/openmpi/lib:$R/ucx/lib:$ROCM/lib:${LD_LIBRARY_PATH:-}
UCX_VER=1.18.0
OMPI_VER=5.0.8
OMPI_TAR=$SW_SRC/openmpi-$OMPI_VER.tar.bz2
B=/scratch/$(id -un)-rochpl-build
LOGS=$R/setup/logs
mkdir -p "$R/src" "$LOGS" "$B"
J=64
say() { echo "[$(date -Iseconds)] $*"; }
if [[ ${REBUILD:-0} == 1 ]]; then
  say "REBUILD=1: removing $R/{ucx,openmpi,rocHPL,BUILD_OK}"
  rm -rf "$R/ucx" "$R/openmpi" "$R/rocHPL" "$R/BUILD_OK" "$B"; mkdir -p "$B"
fi
rm -f "$R/BUILD_OK"
say "build on $(hostname -s), ROCm $ROCM ($(hipconfig --version))"

# ---- UCX ---------------------------------------------------------------------------
if [[ ! -x $R/ucx/bin/ucx_info ]]; then
  say "UCX $UCX_VER"
  cd "$R/src"
  [[ -f ucx-$UCX_VER.tar.gz ]] || curl -fsSLO https://github.com/openucx/ucx/releases/download/v$UCX_VER/ucx-$UCX_VER.tar.gz
  rm -rf "$B/ucx-$UCX_VER"; tar xzf ucx-$UCX_VER.tar.gz -C "$B"
  cd "$B/ucx-$UCX_VER"
  ./contrib/configure-release --prefix="$R/ucx" --with-rocm="$ROCM" --with-verbs --with-rdmacm \
      --without-cuda --without-knem --without-java --enable-mt >"$LOGS/ucx-configure.log" 2>&1
  make -j$J >"$LOGS/ucx-make.log" 2>&1
  make install >"$LOGS/ucx-install.log" 2>&1
fi
"$R/ucx/bin/ucx_info" -v | head -3
"$R/ucx/bin/ucx_info" -d | grep -E 'Transport' | sort | uniq -c
"$R/ucx/bin/ucx_info" -d | grep -q 'Transport: rocm_ipc' || { say "ERROR: UCX has no rocm_ipc"; exit 1; }
ldd "$R/ucx/lib/ucx/libuct_rocm.so" 2>/dev/null | grep -E 'hsa|amdhip'

# ---- Open MPI -----------------------------------------------------------------------
if [[ ! -x $R/openmpi/bin/mpirun ]]; then
  say "Open MPI $OMPI_VER"
  rm -rf "$B/openmpi-$OMPI_VER"; tar xjf "$OMPI_TAR" -C "$B"
  cd "$B/openmpi-$OMPI_VER"
  ./configure --prefix="$R/openmpi" --with-ucx="$R/ucx" --with-rocm="$ROCM" --without-cuda \
      --disable-mpi-fortran --disable-man-pages --enable-mca-no-build=btl-uct >"$LOGS/ompi-configure.log" 2>&1
  make -j$J >"$LOGS/ompi-make.log" 2>&1
  make install >"$LOGS/ompi-install.log" 2>&1
fi
"$R/openmpi/bin/ompi_info" 2>&1 | grep -E 'MCA (pml: ucx|accelerator)'
"$R/openmpi/bin/ompi_info" 2>/dev/null | grep -q 'MCA pml: ucx' || { say "ERROR: Open MPI has no pml ucx"; exit 1; }
"$R/openmpi/bin/ompi_info" 2>/dev/null | grep -q 'MCA accelerator: rocm' || { say "ERROR: Open MPI has no rocm accelerator"; exit 1; }

# ---- rocHPL -------------------------------------------------------------------------
if [[ ! -x $R/rocHPL/bin/rochpl ]]; then
  say "rocHPL $(git -C "$R/src/rocHPL" log -1 --format='%h %ad')"
  rm -rf "$B/rocHPL"; cp -r "$R/src/rocHPL" "$B/rocHPL"   # not cp -a: /scratch cannot keep NFS ACLs
  cd "$B/rocHPL"
  # build for gfx950 only (the CMake default compiles for every arch since gfx900)
  sed -i 's/^set(TARGETS "gfx900;gfx906")/set(TARGETS "gfx950")/; /set(TARGETS "\${TARGETS};/d' CMakeLists.txt
  grep -n 'set(TARGETS' CMakeLists.txt
  mkdir -p build && cd build
  cmake -DCMAKE_INSTALL_PREFIX="$R/rocHPL" -DHPL_MPI_DIR="$R/openmpi" -DROCM_PATH="$ROCM" \
        -DROCBLAS_PATH="$ROCM" -DCMAKE_PREFIX_PATH="$ROCM" -DCMAKE_MODULE_PATH="$ROCM/lib/cmake/hip" \
        -DCMAKE_BUILD_TYPE=Release .. >"$LOGS/rochpl-cmake.log" 2>&1
  make -j$J install >"$LOGS/rochpl-make.log" 2>&1
fi
ls -l "$R/rocHPL/bin"
ldd "$R/rocHPL/bin/rochpl" | grep -E 'rocblas|amdhip|libmpi|ucp'
ldd "$R/rocHPL/bin/rochpl" | grep -E 'rocblas|amdhip' | grep -q rocm-7.14.0 || { say "ERROR: rochpl not linked to ROCm 7.14"; exit 1; }
date -Iseconds >"$R/BUILD_OK"
say "BUILD OK"
