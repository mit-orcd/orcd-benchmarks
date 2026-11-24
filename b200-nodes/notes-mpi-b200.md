# MPI on B200 nodes: suggested admin fix (subnet prefix)

No change to the shared modules (`UCX_NET_DEVICES` is not set there, because the modules are used on all nodes, not only B200).

**Problem:** on B200 nodes, the node-local HCAs `mlx5_0–3` and the 8 NDR HCAs share the default subnet prefix `fe80::`. UCX therefore treats `mlx5_0–3` as able to reach other nodes, and MPI aborts with a UD timeout. Affected: `openmpi/4.1.4`, Spack `openmpi/5.0.8`, and the `nvhpc/26.1` MPI.

**Steps for the admins, on every B200 node:**
1. **Find the subnet manager that runs the node-local network.** It is most likely NVLSM, the NVSwitch management subnet manager. To find it: `systemctl status nvlsm`, `ps -ef | grep -E 'nvlsm|opensm'`, `sminfo -C mlx5_0`.
2. **Set a different prefix in its config file,** for example `subnet_prefix 0xfe80000000000001`. Do not change the cluster fabric's subnet manager (LID 0x719).
3. **Restart that subnet manager,** and Fabric Manager too if NVIDIA requires it.
4. **Check the result.**
   - `/sys/class/infiniband/mlx5_0/ports/1/gids/0` starts with `fe80:0000:0000:0001`.
   - `mlx5_4` still starts with `fe80:0000:0000:0000`.
   - NVLink fabric state is still healthy (`nvidia-smi -q | grep -i fabric`).
5. **Test on two B200 nodes with default module settings,** with no `UCX_NET_DEVICES`: run `osu_latency`/`osu_bw` with `openmpi/4.1.4` and with the `nvhpc/26.1` MPI.
6. **Roll it out to all B200 nodes and make it persistent** in the node image or config management.

**Before starting:** confirm with NVIDIA that a non-default prefix on the NVLSM network is supported, so GPU NVLink is not affected.

**What NVIDIA's Fabric Manager guide says (checked 2026-11-18):**
- `mlx5_0–3` are the 4 PCIe functions of the **CX7 bridge** that connects the NVSwitches to the host on HGX B200: "To manage NVLinks, the NVSwitches are connected to a CX7 Bridge device." Fabric Manager and NVLSM use them to configure NVSwitch/NVLink routing.
- NVLSM config: `nvlsm.conf` in `/usr/share/nvidia/nvswitch/`. Documented options are only `guid`, `daemon`, the gRPC plugin and log options. **`subnet_prefix` is not documented.**
- Warnings: "You can override the contents of nvlsm.conf, but an incorrect override will result a fabric management failure." and "Using infiniband tools over CX interface is not supported."
- Nothing about MPI/UCX using these devices.

So a non-default subnet prefix on the NVLSM network is **not documented as supported**. It needs an NVIDIA support case before production use; a wrong NVLSM config breaks the NVLink fabric (GPUs unusable).

**Option that avoids touching NVLSM:** in the B200 `-c1` node containers, do not expose the CX7 bridge devices to user jobs (`/dev/infiniband/uverbs0–3`, `umad0–3`, `issm0–3`, and `/sys/class/infiniband/mlx5_0–3`). Fabric Manager/NVLSM run outside the job and keep using them; UCX then cannot see `mlx5_0–3` and only finds the NDR HCAs. Needs a test that NVLink still works inside the jobs.

Sources: [NVIDIA Fabric Manager User Guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html), [UCX FAQ](https://openucx.readthedocs.io/en/master/faq.html).

**Not fixed by this (separate module problems):**
- Spack `openmpi/5.0.8` also fails in `MPI_Init` with a PMIx error; `PMIX_MCA_gds=hash` works around it.
- The core `openmpi/5.0.8` cannot start because `libhcoll` is not installed on the B200 nodes.

Details and test results: `out-osu-ib/admin-subnet-prefix-fix.md`, `out-osu-ib/ompi-5.0.8/report.md`, `out-osu-ib/ucx-net-devices/report.md`, `out-osu-ib/nvhpc-26.1/report.md`, `out-osu-ib/nvhpc-26.1/devices.md`.
