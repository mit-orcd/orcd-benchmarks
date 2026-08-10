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

*Update (2026-08-10, after the 2-node runs):* the Ubuntu nodes turn out to boot
the **same** `iommu=pt intel_iommu=on` with the same 540 IOMMU groups, and they
still reach full inter-node bandwidth. So IOMMU is not what separates the two
clusters — see the configuration comparison below.

---

# Configuration differences: Ubuntu nodes vs Rocky 8 nodes

Measured on node5700/node5701 on 2026-08-10, alongside the 2-node NCCL runs.
The question this answers: *are the IB driver and system configuration the same
on the two clusters?* **No.**

| Item | Ubuntu (node5700/5701) | Rocky 8 (node5500-5502) | same? |
|------|------------------------|-------------------------|-------|
| IOMMU (kernel cmdline) | `iommu=pt intel_iommu=on`, 540 groups | `iommu=pt intel_iommu=on`, 540 groups | **same** |
| NCCL | 2.29.2 | 2.29.2 | **same** |
| GPUDirect RDMA | `nvidia_peermem` loaded, DMABUF path | `nvidia_peermem` loaded, DMABUF path | **same** |
| IB rails | 8 x 400 Gb/s NDR, MTU 4096 | 8 x 400 Gb/s NDR | **same** |
| **MOFED / rdma-core** | OFED-internal-**25.10**-1.7.1.413 | OFED-internal-**26.04**-0.8.6 | **differs** |
| **NVIDIA driver** | **570.211.01** | **590.48.01** | **differs** |
| **Kernel** | 6.8.0-124 on both nodes | **4.18** (5500) / **6.12** (5502) — heterogeneous | **differs** |
| **CUDA (build)** | 12.9 | 13.1 | **differs** |
| PCI cmdline | `pci=realloc=off` | `pci=disable_acs_redir=...` on 5502 only | differs |
| CPU / governor | Xeon Platinum 8570, `performance` | not verifiable from here | unknown |
| HCA firmware | 28.47.2526 | not verifiable from here | unknown |

## Why this matters

The 2-node NCCL runs show the Ubuntu pair is **1.9-3.0x faster per operation** on
small messages for every collective that NCCL splits across its 8 channels,
while `sendrecv` and `gather` — the two it does not split — are **identical** on
both clusters at every message size. The cost is therefore paid *per network
operation*, not per byte. Full analysis: `out-nccl-2node/summary.md`, section 4,
"Why small messages favour the Ubuntu nodes".

**The IOMMU hypothesis is retired.** Both clusters boot identical IOMMU settings,
so IOTLB pressure cannot be the differentiator. (The `iommu=off` advice in
`../b200-nodes/notes.md` concerned a different problem — the bulk GPU-read
bandwidth cap — and is unrelated to this per-operation gap.)

**Live candidates**, in order of suspicion:

1. **The InfiniBand stack** (MOFED 25.10 here vs 26.04 there). The verbs
   provider is exactly the layer that sets per-operation posting cost while
   leaving bulk streaming untouched — which is the precise shape of the
   measurement.
2. **GPU driver / CUDA pair** (570 + CUDA 12.9 vs 590 + CUDA 13.1).
3. **Host CPU cost in NCCL's proxy thread**, which posts each RDMA operation and
   so scales with operation *count* rather than bytes. A different CPU or a
   `powersave` governor on the Rocky side would produce this signature.

Note the counter-intuitive direction: **Rocky 8 runs the newer MOFED and the
newer driver, yet is slower per operation** — consistent with a regression in the
newer IB stack, though not proven by this data.

## Unverified, and how to settle it

CPU model, frequency governor and HCA firmware could **not** be compared: the
Rocky nodes are Slurm-managed and refuse ssh from node5700
(`Host key verification failed`), so those rows rest on `../b200-nodes/notes.md`
and the run logs. Any of the three could matter for a per-operation cost.

The decisive test is an **`ib_write_bw` small-message sweep** — many small
operations versus one large one — run on both clusters. That separates the IB
stack from everything above it. It needs a Slurm job on the Rocky side, which
cannot be launched from node5700.

Node heterogeneity is *not* the explanation: all three Rocky pairs
(5500+5501, 5500+5502, 5501+5502) show the same slow small-message times, so it
is systematic rather than one degraded node.
