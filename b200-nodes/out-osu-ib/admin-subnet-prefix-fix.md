# B200 nodes: MPI over InfiniBand fails — suggested admin fix (subnet prefix)

**Problem.** Each B200 node has 8 NDR HCAs on the cluster fabric
(`mlx5_4/7/8/9/10/13/14/15`, SM LID 0x719) plus 4 HCAs on a node-local network
(`mlx5_0-3`, 100 Gb/s, its own SM at LID 0x1, most likely the NVLink/NVSwitch
management network run by NVLSM). Both networks use the default subnet prefix
`fe80:0000:0000:0000`. UCX treats ports with the same prefix as reachable, uses
`mlx5_0-3` for inter-node traffic, and MPI aborts with
`ud_ep.c: Fatal: UD endpoint ... unhandled timeout error`.
Affected: `openmpi/4.1.4`, Spack `openmpi/5.0.8`, and the HPC-X OpenMPI in
`nvhpc/26.1` (host-memory traffic). L40S and H200 nodes are not affected: all
their active HCAs are on the cluster fabric.

**Suggested fix (no change to the shared modules):** give the node-local
network its own subnet prefix on every B200 node.

1. On one B200 node, find the SM that runs the node-local network
   (e.g. `systemctl status nvlsm`, `ps -ef | grep -E 'nvlsm|opensm'`,
   `sminfo -C mlx5_0`), and its config file.
2. In that SM's config, set a non-default prefix, for example
   `subnet_prefix 0xfe80000000000001`.
   Do not change the cluster fabric SM (LID 0x719).
3. Restart that SM (and Fabric Manager if NVIDIA requires it), so the
   `mlx5_0-3` ports get the new prefix.
4. Check: `cat /sys/class/infiniband/mlx5_0/ports/1/gids/0` starts with
   `fe80:0000:0000:0001`; `mlx5_4` still starts with `fe80:0000:0000:0000`;
   `nvidia-smi -q | grep -i fabric` shows NVLink fabric state Completed.
5. Test on two B200 nodes with default module settings (no `UCX_NET_DEVICES`):
   `osu_latency` / `osu_bw` with `openmpi/4.1.4` and nvhpc/26.1 MPI.
6. Apply to all B200 nodes and make it persistent (node image / config
   management), since the change is per node.

**Before changing it:** confirm with NVIDIA that a non-default subnet prefix
on the NVLSM network is supported, so GPU NVLink is not affected.

**Not fixed by this (separate issues):**
- Spack `openmpi/5.0.8`: PMIx `gds/shmem2` error in `MPI_Init`
  (workaround `PMIX_MCA_gds=hash`).
- core `openmpi/5.0.8`: needs `libhcoll.so.1` / `libocoms.so.0`
  (`/opt/mellanox/hcoll`), not present on B200 nodes.

Evidence: `out-osu-ib/ompi-5.0.8/report.md`, `out-osu-ib/ucx-net-devices/report.md`,
`out-osu-ib/nvhpc-26.1/report.md`, `out-osu-ib/nvhpc-26.1/devices.md`.
