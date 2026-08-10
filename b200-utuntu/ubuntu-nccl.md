# Two-node NCCL on node5700 + node5701 — what is needed

Question: to run 2-node NCCL between the Ubuntu nodes **node5700** and
**node5701**, is passwordless SSH required, and what else? (Assuming **Slurm is
not set up** on these nodes.) This is the list to hand to the admins.

---

## Yes — passwordless SSH is required

Without Slurm, `mpirun` has no other way to start ranks on the second node: it
shells out to `ssh node5701` and launches `orted` there. The same applies to the
Megatron 2-node case (torchrun rendezvous), unless you are willing to hand-start
a process in two terminals.

**Specifics:** node-to-node, **both directions**, non-interactive, for the normal
user. `$HOME` is **local disk** on these nodes (only `/orcd/data/orcd/022` is
NFS), so a key installed on node5700 does *not* appear on node5701 — the admin
either installs the key in each node's `~/.ssh/authorized_keys`, sets up
host-based authentication, or mounts a shared home. `StrictHostKeyChecking` must
also be satisfied (pre-seeded `known_hosts`) so the first connection does not
prompt.

## The rest of the list

1. **`memlock` unlimited** (`ulimit -l`) for the user on both nodes, including in
   non-interactive SSH sessions — `/etc/security/limits.conf` plus `UsePAM yes`.
   This is the single most common cause of "InfiniBand works interactively but
   fails under mpirun". InfiniBand pins memory; a low limit makes NCCL silently
   fall back to TCP or fail outright.

2. **No firewall between the nodes** on the inband network. Open MPI's
   out-of-band channel and NCCL's bootstrap open arbitrary TCP ports; if a
   firewall is present they need a permitted range
   (`--mca oob_tcp_dynamic_ipv4_ports`, `NCCL_PORT_RANGE`).

3. **Same IB fabric state on node5701** as node5700: the 8 NDR rails
   (`mlx5_4,7,8,9,10,13,14,15`) Active at 400 Gb/s, `ibstat` clean, and the
   subnet manager seeing both nodes. Confirm the HCA **names** match — the NCCL
   scripts pin `NCCL_IB_HCA` by name.

4. **rdma-core / libibverbs userspace** on node5701 (`libibverbs1`,
   `librdmacm1`, `ibverbs-providers`). Present on node5700; HPC-X's bundled UCX
   links against the system libraries.

5. **GPUDirect RDMA**: `nvidia_peermem` loaded, or a driver + MOFED combination
   new enough for the DMABUF path. On the Rocky 8 nodes NCCL selected DMABUF and
   worked, so this is more about verifying than installing.

6. **`perftest`** (`ib_write_bw`) on both nodes — not needed to run NCCL, but it
   is how you tell a bad GPUDirect path from a bad NCCL configuration. The Rocky
   nodes had a ~18.5 GB/s NIC-reads-from-GPU cap that limited 2-node NCCL, and
   `ib_write_bw --use_cuda` is what diagnosed it (see
   `../b200-nodes/notes.md`). Worth requesting while the admins are in there.

7. **`/orcd/data/orcd/022` mounted at the identical path on node5701** — that is
   where NVHPC, the nccl-tests binaries and the run scripts live. Same path on
   both nodes, or `mpirun` cannot find the binary.

8. **Matching node5701 image**: same driver (570.211.01), same kernel/OS. A
   driver mismatch across nodes will break NCCL.

## Also worth raising with the admins

Ask whether **IOMMU** is enabled (`iommu=pt intel_iommu=on`). On the Rocky 8
B200 nodes that was the prime suspect for the GPU-read bandwidth cap, and the
reference cluster required `iommu=off` to reach full P2P throughput. It is a
reboot-level change, so better raised now than after a disappointing 2-node
number.
