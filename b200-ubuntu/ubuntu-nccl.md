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

**Nothing in this table needs to be changed for the 2-node NCCL test to be valid
or for the results already collected to stand.** The last column says what, if
anything, to ask for.

| Item | Ubuntu (node5700/5701) | Rocky 8 (node5500-5502) | same? | Reconfigure? |
|------|------------------------|-------------------------|-------|--------------|
| IOMMU (kernel cmdline) | `iommu=pt intel_iommu=on`, 540 groups | `iommu=pt intel_iommu=on`, 540 groups | **same** | **No — on these Ubuntu nodes.** They already reach full GPUDirect line rate *with* IOMMU on, so there is nothing to recover here. On the **Rocky 8** nodes `iommu=off` is still worth trying as a one-node diagnostic — see "Problem 1" below. |
| NCCL | 2.29.2 | 2.29.2 | **same** | No. |
| GPUDirect RDMA | `nvidia_peermem` loaded, DMABUF path | `nvidia_peermem` loaded, DMABUF path | **same** | No. |
| IB rails | 8 x 400 Gb/s NDR, MTU 4096 | 8 x 400 Gb/s NDR | **same** | No. |
| **MOFED / rdma-core** | OFED-internal-**25.10**-1.7.1.413 | OFED-internal-**26.04**-0.8.6 | **differs** | **Not required.** Only if you want to test the leading hypothesis for the small-message gap, or for fleet consistency later. Change one cluster at a time or the experiment tells you nothing. |
| **NVIDIA driver** | **570.211.01** | **590.48.01** | **differs** | **Not required — and disruptive.** An upgrade to r580+ would invalidate every result collected so far and force a rebuild against CUDA 13. Only worth it if these nodes must match the production fleet. |
| **Kernel** | 6.8.0-124 on both nodes | **4.18** (5500) / **6.12** (5502) — heterogeneous | **differs** | **No.** The Rocky pair is itself heterogeneous and NCCL runs fine across it, so this class of difference is demonstrably tolerable. |
| **CUDA (build)** | 12.9 | 13.1 | **differs** | **No — and not possible** without the driver upgrade above (CUDA 13 needs r580+). |
| PCI cmdline | `pci=realloc=off` | `pci=disable_acs_redir=...` on 5502 only | differs | No. Not on the NCCL data path here. |
| CPU / governor | Xeon Platinum 8570, `performance` | not verifiable from here | unknown | **Ask them to report it, not change it.** NCCL's proxy thread posts every RDMA operation on the host CPU, so a `powersave` governor on the Rocky side would explain the per-operation gap. Read-only check. |
| HCA firmware | 28.47.2526 | not verifiable from here | unknown | **Ask them to report it, not change it.** |

## What to actually ask the admins for

**Required — none.** Both Ubuntu nodes are already internally consistent (same
driver, MOFED, kernel, firmware, CPU, governor), which is the only consistency
NCCL needs: the two nodes *within a job* must agree, and they do. `memlock` is
already unlimited and passwordless ssh is in place.

**Worth asking for (cheap, read-only):**

1. CPU model + `scaling_governor` on node5500-5502.
2. HCA firmware (`ibv_devinfo | grep fw_ver`) on node5500-5502.
3. Shell access to one Rocky node, or a Slurm job slot, so the `ib_write_bw`
   small-message sweep can be run there — the one test that would settle the
   cause.

**Optional cleanup (not a blocker):** remove the Debian/Ubuntu default line
`127.0.1.1 node57xx` from `/etc/hosts` on node5700/node5701, or point it at the
real inband address. It makes each node resolve its own name to loopback, which
hangs Open MPI in `MPI_Init`. `run-nccl-2node.sh` already works around it by
pinning `--mca btl_tcp_if_include eno3`, so this only spares the next person the
debugging.

**Do not ask for (on node5700/node5701):** `iommu=off`, driver/MOFED/kernel
alignment, or CUDA 13 — none of it is needed for this benchmark, and the driver
change would cost you the current results. This is scoped to the *Ubuntu* nodes:
the Rocky 8 nodes have real deficits and their own action list below.

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

## GPUDirect RDMA measured directly (2026-08-10)

`ib_write_bw` on mlx5_4, 64 MiB RDMA writes, 200 iterations, node5700 <-> node5701.
The Rocky 8 column is the same test from `../b200-nodes/notes.md` (2026-07-13).

| Test | **Ubuntu 5700<->5701** | Rocky 8 5500<->5502 | ratio |
|------|----------------------:|--------------------:|------:|
| host mem -> host mem | 378.5 Gb/s | 379.5 Gb/s | 1.00x |
| **NIC reads from GPU** | **395.5 Gb/s** | 147.6 Gb/s | **2.68x** |
| **NIC writes into GPU** | **379.6 Gb/s** | 286.6 Gb/s | **1.32x** |

Reproduce:

```bash
# server (node5701); add --use_cuda=0 to test NIC *writes into* GPU
ssh node5701 'ib_write_bw -d mlx5_4 --report_gbits -s 67108864 -n 200'
# client (node5700); --use_cuda=0 makes the NIC *read from* GPU memory
ib_write_bw -d mlx5_4 --use_cuda=0 --report_gbits -s 67108864 -n 200 node5701
```

**The Ubuntu nodes run GPUDirect at full line rate in both directions**, while
booting the *identical* `iommu=pt intel_iommu=on` with the same 540 IOMMU groups
as the Rocky 8 nodes.

**What this does and does not show about IOMMU/ACS.** The classic mechanism is
real: Linux enables ACS on downstream ports when the IOMMU is on, ACS redirect
routes peer-to-peer TLPs up to the root complex instead of straight across the
switch, and that can throttle GPUDirect — which is why the reference cluster
needed `iommu=off`.

The measurement above shows something narrower: **IOMMU-on is not *inherently*
fatal on this hardware.** node5700 boots `iommu=pt intel_iommu=on`, places the
HCA (`0000:18:00.0`, IOMMU group 62) and GPU0 (`0000:1b:00.0`, group 65) in
separate groups — so ACS isolation is active — and still reads from GPU at
**395.5 Gb/s**, NDR line rate with nothing left to recover. `nvidia-smi topo -m`
reports **PXB** for every GPU<->rail pair (across PCIe bridges, *not* via the
host bridge) and `lspci -t` confirms the HCA and GPU sit under the same switch.

So `iommu=off` is **not** ruled out for the Rocky 8 nodes — it remains a
legitimate, cheap, reversible **diagnostic** there. What is ruled out is the
assumption that IOMMU-on *must* cap GPUDirect: it does not on this platform, so
if it does on theirs, the difference is in ACS state or PCIe topology, and that
is worth identifying rather than papering over.

(An earlier revision of this file said "do not spend a reboot on `iommu=off`".
That was too strong and has been corrected.)

# Suggestions for the admins

There are **two separate deficits** on the Rocky 8 nodes; they need different
fixes. node5700/node5701 now serve as a known-good reference to diff against.

## Problem 1 — the GPUDirect bulk cap (largest, 2.7x)

This is what holds Rocky 8 to 12.7 GB/s on 1-GPU/node NCCL where the Ubuntu pair
reaches 48.7 GB/s.

1. **Re-run the perftest triplet above on Rocky 8** to confirm the cap still
   exists — those numbers date from 2026-07-13 and the stack has moved since.
1b. **Boot one Rocky node with `iommu=off` and re-run it.** Decisive: if
   NIC-reads-from-GPU jumps from 147.6 Gb/s toward ~395, the IOMMU/ACS
   interaction on that platform is the cause. If it does not move, IOMMU is
   exonerated there too and the search moves to BIOS/firmware and the stack.
   Prefer a **targeted** fix in production if this proves the point: disabling
   ACS redirect on the relevant ports (BIOS ACS, or `pci=disable_acs_redir=`
   with the *correct* device IDs) keeps IOMMU isolation. Note node5502 already
   carries `pci=disable_acs_redir=pci:1000:c030` and was still capped — more
   likely that mask missed the Broadcom switch ports in its path than that ACS
   is innocent.
2. **Diff the PCIe configuration against node5700** (root needed). Start with
   `nvidia-smi topo -m`: node5700 reports **PXB** for each GPU<->rail pair. If a
   Rocky node reports `NODE`/`SYS` instead, the GPU and its rail are not under a
   common switch there and that topology difference alone could explain the cap.
   Then `lspci -vvv`
   **ACS control bits** on the Broadcom switches, **PCIe Relaxed Ordering**, and
   **Max Payload Size**. Relaxed Ordering is the first thing to check — it is an
   NVIDIA-recommended BIOS setting for GPUDirect and disabling it degrades
   NIC-reads-from-GPU specifically, matching the read/write asymmetry above.
   Note node5700 does **not** carry `pci=disable_acs_redir` and is still at full
   speed, so the kernel ACS workaround is not the differentiator either; this
   points at BIOS/firmware level.
3. **A/B the software stack on one Rocky node**: install MOFED 25.10 and driver
   570.211.01 to match the Ubuntu nodes, then re-run perftest. Cheap, reversible,
   and separates software stack from platform configuration.

## Problem 2 — the per-operation small-message cost (2-3x)

Affects every collective NCCL splits across its 8 channels; `sendrecv` and
`gather` are unaffected. Cheapest checks first, all read-only or reversible:

1. **CPU frequency governor and C-states** on node5500-5502. NCCL's proxy thread
   posts every RDMA operation on the host CPU, so `powersave` or deep C-states
   produce exactly this signature. The Ubuntu nodes run `performance`. Zero risk
   to set.
2. **`NCCL_DEBUG=INFO` on both clusters**, comparing channel count, protocol
   (LL / LL128 / Simple) and algorithm. If Rocky 8 selects a different protocol
   this is a tuning issue fixable with environment variables, not a
   reconfiguration.
3. **Align node5500's kernel** (EL8 / 4.18) with node5502 (EL10 / 6.12). Not the
   sole cause — all three Rocky pairs are slow — but running a 4.18 kernel under
   MOFED 26.04 is worth removing as a variable.

## What is established, and what is not

Established by measurement: the cause is **not** IOMMU, **not** the NCCL version
(2.29.2 on both), **not** node heterogeneity (all three Rocky pairs behave the
same), and **not** bulk fabric health (host-to-host is 378-380 Gb/s on both).

Not established: the actual root cause of either deficit. Step 2 of Problem 1 —
the PCIe/BIOS diff against node5700 — is where the effort is best spent, since it
is the largest deficit and now has a working reference to compare against.

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
