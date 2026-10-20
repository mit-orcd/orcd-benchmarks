# rocHPL for MI355X — installation

rocHPL (github.com/ROCm/rocHPL) is AMD's HPL (fp64 Linpack) for ROCm GPUs. It needs a
GPU-aware MPI: rocHPL passes GPU buffers straight to MPI calls. It does not use RCCL.

Built on node6100 (Ubuntu 24.04, 8 x MI355X gfx950) with ROCm 7.14 from
`../amd-software/rocm-7.14.0` (`ROCM_STACK=7.14`), not the host /opt/rocm 7.2.4. Both AMD nodes
see the same tree, so node6101 uses the same build.

| Part | Version | Built with | Install dir |
|---|---|---|---|
| UCX | 1.18.0 | `--with-rocm --with-verbs --with-rdmacm --enable-mt` | `ucx/` |
| Open MPI | 5.0.8 | `--with-ucx --with-rocm` (pml ucx, rocm accelerator) | `openmpi/` |
| rocHPL | git main | ROCm 7.14 hipcc + rocBLAS, gfx950 only, Release | `rocHPL/` |

The Open MPI in `../amd-software/openmpi` is built without UCX and ROCm (it only launches
rccl-tests), so rocHPL uses this separate build.

## Build

```bash
ssh node6100.inband 'bash /orcd/data/orcd/022/benchmarks/rochpl/setup/build-rochpl.sh \
    > /orcd/data/orcd/022/benchmarks/rochpl/setup/logs/build.log 2>&1'
```

- Builds in node-local `/scratch/$USER-rochpl-build` (NFS builds are much slower).
- Each step is skipped if its install already exists; `REBUILD=1` wipes them first.
  Writes `BUILD_OK` at the end, after checking that `rochpl` links to ROCm 7.14.
- Logs: `setup/logs/`.

Check that UCX sees the GPU and the NICs:

```bash
ucx/bin/ucx_info -d | grep Transport | sort | uniq -c   # rocm_copy, rocm_ipc, rc_verbs (ionic_*)
openmpi/bin/ompi_info | grep -E 'pml: ucx|accelerator: rocm'
```

## Run

Run scripts are in `../amd-ubuntu/rochpl/`:

- `run_rochpl.sh <G>`: one run on G GPUs (1–8 on one node, 16 on two nodes).
- `run_rochpl_all.sh`: the full sweep, then the analysis and push. Start it from a login node
  with `../amd-ubuntu/launch.sh node6100 rochpl/run_rochpl_all.sh`.

Results: `../amd-ubuntu/results/rochpl-vs-rvs.md`.
