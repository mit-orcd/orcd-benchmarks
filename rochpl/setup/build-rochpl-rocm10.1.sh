#!/usr/bin/env bash
# Build rocHPL for MI355X (gfx950) with a GPU-aware MPI against ROCm 10.1.0. Twin of
# build-rochpl.sh (ROCm 7.14); same UCX 1.18.0, Open MPI 5.0.8 and rocHPL commit, new prefixes only:
#   UCX + Open MPI  -> amd-software/ompi-5.0.8-rocm10.1/{ucx,openmpi}
#   rocHPL          -> rochpl/build-rocm10.1/rocHPL
# The ROCm 7.14 build in rochpl/{ucx,openmpi,rocHPL} is not touched.
# Called by amd-software/setup/install-rocm-10.1.sh on an AMD node. Writes build-rocm10.1/BUILD_OK.
set -euo pipefail
export ROCM_STACK=10.1
source /orcd/data/orcd/022/benchmarks/amd-software/env.sh
R=/orcd/data/orcd/022/benchmarks/rochpl
P=$SW/ompi-5.0.8-rocm10.1
H=$R/build-rocm10.1
ROCM=$ROCM_PATH
export PATH=$ROCM/bin:$PATH
export LD_LIBRARY_PATH=$P/openmpi/lib:$P/ucx/lib:$ROCM/lib:${LD_LIBRARY_PATH:-}
UCX_VER=1.18.0
OMPI_VER=5.0.8
OMPI_TAR=$SW_SRC/openmpi-$OMPI_VER.tar.bz2
B=/scratch/$(id -un)-rochpl-build-rocm10.1
LOGS=$R/setup/logs/rocm10.1
mkdir -p "$P" "$H" "$LOGS" "$B"
J=32
say() { echo "[$(date -Iseconds)] $*"; }
rm -f "$H/BUILD_OK"
say "build on $(hostname -s), ROCm $ROCM ($(hipconfig --version))"

# ---- UCX ---------------------------------------------------------------------------
if [[ ! -x $P/ucx/bin/ucx_info ]]; then
  say "UCX $UCX_VER"
  rm -rf "$B/ucx-$UCX_VER"; tar xzf "$R/src/ucx-$UCX_VER.tar.gz" -C "$B"
  cd "$B/ucx-$UCX_VER"
  nice -n 19 ./contrib/configure-release --prefix="$P/ucx" --with-rocm="$ROCM" --with-verbs --with-rdmacm \
      --without-cuda --without-knem --without-java --enable-mt >"$LOGS/ucx-configure.log" 2>&1
  nice -n 19 make -j$J >"$LOGS/ucx-make.log" 2>&1
  make install >"$LOGS/ucx-install.log" 2>&1
fi
"$P/ucx/bin/ucx_info" -v | head -3
"$P/ucx/bin/ucx_info" -d | grep -q 'Transport: rocm_ipc' || { say "ERROR: UCX has no rocm_ipc"; exit 1; }

# ---- Open MPI -----------------------------------------------------------------------
if [[ ! -x $P/openmpi/bin/mpirun ]]; then
  say "Open MPI $OMPI_VER"
  rm -rf "$B/openmpi-$OMPI_VER"; tar xjf "$OMPI_TAR" -C "$B"
  cd "$B/openmpi-$OMPI_VER"
  nice -n 19 ./configure --prefix="$P/openmpi" --with-ucx="$P/ucx" --with-rocm="$ROCM" --without-cuda \
      --disable-mpi-fortran --disable-man-pages --enable-mca-no-build=btl-uct >"$LOGS/ompi-configure.log" 2>&1
  nice -n 19 make -j$J >"$LOGS/ompi-make.log" 2>&1
  make install >"$LOGS/ompi-install.log" 2>&1
fi
"$P/openmpi/bin/ompi_info" 2>/dev/null | grep -q 'MCA pml: ucx' || { say "ERROR: Open MPI has no pml ucx"; exit 1; }
"$P/openmpi/bin/ompi_info" 2>/dev/null | grep -q 'MCA accelerator: rocm' || { say "ERROR: Open MPI has no rocm accelerator"; exit 1; }

# ---- rocHPL (same source tree and commit as the 7.14 build) ---------------------------
if [[ ! -x $H/rocHPL/bin/rochpl ]]; then
  say "rocHPL $(git -C "$R/src/rocHPL" log -1 --format='%h %ad')"
  rm -rf "$B/rocHPL"; cp -r "$R/src/rocHPL" "$B/rocHPL"
  cd "$B/rocHPL"
  sed -i 's/^set(TARGETS "gfx900;gfx906")/set(TARGETS "gfx950")/; /set(TARGETS "\${TARGETS};/d' CMakeLists.txt
  rm -rf build; mkdir -p build && cd build
  nice -n 19 cmake -DCMAKE_INSTALL_PREFIX="$H/rocHPL" -DHPL_MPI_DIR="$P/openmpi" -DROCM_PATH="$ROCM" \
        -DROCBLAS_PATH="$ROCM" -DCMAKE_PREFIX_PATH="$ROCM" -DCMAKE_MODULE_PATH="$ROCM/lib/cmake/hip" \
        -DCMAKE_BUILD_TYPE=Release .. >"$LOGS/rochpl-cmake.log" 2>&1
  nice -n 19 make -j$J install >"$LOGS/rochpl-make.log" 2>&1
fi
ls -l "$H/rocHPL/bin"
ldd "$H/rocHPL/bin/rochpl" | grep -E 'rocblas|amdhip|libmpi|ucp' || true
ldd "$H/rocHPL/bin/rochpl" | grep -E 'rocblas|amdhip' | grep -q rocm-10.1.0 || { say "ERROR: rochpl not linked to ROCm 10.1"; exit 1; }
rm -rf "$B"
date -Iseconds >"$H/BUILD_OK"
say "BUILD OK"
