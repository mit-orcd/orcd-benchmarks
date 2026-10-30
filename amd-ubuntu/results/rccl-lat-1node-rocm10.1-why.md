# Why RCCL latency inside one node changed from ROCm 7.14 to 10.1 (MI355X)

node6100, 8× MI355X, one node (XGMI only). Run dir: `logs/node6100/rocm10.1/rccl/why_1node_20261029_143522`. Script: [`../rocm10.1/run_rccl_why_1node.sh`](../rocm10.1/run_rccl_why_1node.sh). Question from [`rocm10.1-vs-rocm7.14.md`](rocm10.1-vs-rocm7.14.md) §5.1: are the 20-37% changes noise between runs, or did RCCL change how it runs these collectives?

## Answer

**Main factors for small-message latency in one node:** (1) fixed cost per call (kernel launch, set-up, final sync), which dominates at 8 B-1 KB; (2) algorithm (ring, tree, direct), which sets how many steps and GPU-to-GPU handshakes a call needs; (3) protocol (LL, LL128, Simple); (4) number of channels; (5) implementation path (copy through RCCL buffers, or read peer GPU memory directly); (6) the tuning table that picks algorithm and protocol per size.

**Same in 7.14 and 10.1** (debug logs): channels (112 / 64 / 8), algorithm and protocol where logged (e.g. broadcast RING/LL up to 64K on both), and the tuning table (the new 10.1 rows are all_reduce at 32 MB and above).

**What differs and why:**
- **Implementation path (5):** RCCL 2.30.7 adds a new DDA-IPC path for all_reduce and all_gather. Each GPU reads the other 7 GPUs' buffers directly in one step, instead of passing data around a ring in 7 steps with a handshake each. Fewer steps, less waiting: this is the likely reason all_gather is about 30% faster and all_reduce 1M about 20% faster.
- **Fixed cost per call (1), probably:** sendrecv 1K and broadcast 1K are about 30% slower. Their algorithm, protocol and channels are the same, and the change is the same with DDA on or off, so it points to a higher per-call cost in RCCL 2.30.7 or HIP 7.16 (a deduction, not measured directly).
- **Not explained:** alltoall is about 30% faster without using DDA, so it comes from another part of the 2.30.7 rework; the logs do not say which.

**Notes:** this is not noise (repeat runs agree within about ±3%; the changes are 20-37%). `RCCL_DDA_ENABLE=0` does not restore 7.14: it falls back to a path slower than both versions (12 of 21 points more than 10% slower), so it cannot show how much DDA explains.

Noise between runs is small: the three default runs per version in §5.1 agree within about ±3% on both nodes (node6100 and node6101), while the changes are 20-37%.

## 1. Timing, same day (time per call, µs, out-of-place, lower is better)

rccl-tests, 1000 iterations + 100 warm-up, no data check, NUMA-bound ranks; one run per column, back to back. Change = (10.1 − 7.14) / 7.14, so a negative number means 10.1 is faster.

| Collective | Size | DDA on 10.1 | 7.14 | 10.1 | 10.1, `RCCL_DDA_ENABLE=0` | 10.1 vs 7.14 | DDA off vs 7.14 |
|---|---:|---|---:|---:|---:|---:|---:|
| sendrecv | 8 | no | 7.7 | 7.0 | 7.3 | -10.2% | -5.6% |
| sendrecv | 1K | no | 6.5 | 8.4 | 8.5 | +30.0% | +31.4% |
| sendrecv | 64K | no | 13.4 | 13.9 | 13.9 | +4.0% | +4.0% |
| sendrecv | 1M | no | 29.6 | 30.1 | 30.0 | +1.6% | +1.4% |
| all_reduce | 8 | yes | 14.6 | 15.2 | 16.6 | +4.4% | +13.3% |
| all_reduce | 1K | yes | 4.9 | 5.0 | 18.1 | +2.3% | +272.1% |
| all_reduce | 64K | yes | 6.4 | 6.3 | 23.4 | -1.4% | +263.8% |
| all_reduce | 1M | yes | 17.4 | 13.7 | 35.1 | -21.2% | +101.4% |
| all_gather | 1K | yes | 7.8 | 5.0 | 13.4 | -35.5% | +72.7% |
| all_gather | 64K | yes | 9.0 | 6.3 | 14.7 | -30.0% | +63.8% |
| all_gather | 1M | yes | 11.7 | 7.9 | 30.5 | -32.7% | +159.8% |
| reduce_scatter | 1K | not checked | 6.1 | 6.3 | 14.0 | +3.6% | +130.6% |
| reduce_scatter | 64K | not checked | 7.1 | 7.1 | 14.8 | 0.0% | +107.7% |
| reduce_scatter | 1M | not checked | 8.8 | 8.9 | 16.1 | +0.9% | +83.1% |
| broadcast | 8 | no | 7.0 | 6.1 | 7.8 | -12.2% | +11.2% |
| broadcast | 1K | no | 5.8 | 7.7 | 8.0 | +32.9% | +37.1% |
| broadcast | 64K | no | 15.0 | 15.6 | 15.6 | +3.8% | +3.6% |
| broadcast | 1M | no | 41.7 | 42.4 | 42.6 | +1.8% | +2.3% |
| alltoall | 1K | no | 7.5 | 4.9 | 9.2 | -34.8% | +22.8% |
| alltoall | 64K | no | 10.4 | 7.6 | 11.2 | -27.4% | +7.2% |
| alltoall | 1M | no | 14.0 | 9.2 | 15.7 | -34.0% | +12.0% |

## 2. What RCCL picks (debug runs, rank 0)

`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL,ENV`, 8 B to 1 MB, 5 iterations; separate from the timing runs because debug output slows every call.

| Collective | Stack | Channels (coll / p2p / p2p per peer) | Implementation lines | DDA path by size | Algo/proto lines |
|---|---|---|---|---|---|
| sendrecv | 7.14 | 112 / 64 / 8 | none | none | not logged |
| sendrecv | 10.1 | 112 / 64 / 8 | none | none | not logged |
| all_reduce | 7.14 | 112 / 64 / 8 | none | none | 8: TREE/LL |
| all_reduce | 10.1 | 112 / 64 / 8 | DDA-IPC ×205 | yes, sizes not logged | 8: TREE/LL |
| all_gather | 7.14 | 112 / 64 / 8 | none | none | not logged |
| all_gather | 10.1 | 112 / 64 / 8 | DDA-IPC ×169 | yes, sizes not logged | not logged |
| broadcast | 7.14 | 112 / 64 / 8 | none | none | 8: RING/LL; 1K: RING/LL; 64K: RING/LL; 1M: RING/SIMPLE |
| broadcast | 10.1 | 112 / 64 / 8 | none | none | 8: RING/LL; 1K: RING/LL; 64K: RING/LL; 1M: RING/SIMPLE |
| alltoall | 7.14 | 112 / 64 / 8 | none | none | not logged |
| alltoall | 10.1 | 112 / 64 / 8 | none | none | not logged |

Both versions load RCCL's gfx950 tuning table (`share/rccl/tuner/rccl_tuner_gfx950.csv`): 4 rows on 7.14, 7 on 10.1. The rows only in 10.1 are all_reduce at 32-52 MB, 52-112 MB, 112-224 MB, far above the 1 MB used here, so the table does not explain these changes.

`RCCL_DDA_ENABLE` and related settings (`RCCL_DDA_THRESHOLD`, `RCCL_DDA_LL_*`) exist in RCCL 2.30.7 (strings in `librccl.so.1`); 7.14's RCCL 2.30.4 logs no DDA lines (none of the collectives).

## 3. Debug lines only one version prints (rank 0, numbers replaced by #)

### sendrecv

Only in 10.1:
```
  218  Launch P#P kernel: gridDim.x=# blockDim.x=# (p#pnChannels=#, p#pnChannelsPerPeer=#, nChannels=#, nWorkBatches=#)
  121  RCCL P#P SendRecv protocol=LL dir=recv bytes=# nChannels=#
  121  RCCL P#P SendRecv protocol=LL dir=send bytes=# nChannels=#
   97  RCCL P#P SendRecv protocol=Simple dir=recv bytes=# nChannels=#
   97  RCCL P#P SendRecv protocol=Simple dir=send bytes=# nChannels=#
    2  Found peer memory client /sys/memory_peers/amdkfd
    1  Socket handshake magic #x# (built-in default)
    1  cuMem auto-enable is limited to gfx# (detected gfx#); set NCCL_CUMEM_ENABLE=# to override
    1  RCCL version : #.#.#-HEAD:#da#
    1  NET/IB : IB Use Inline is disabled and CTS Offload is enabled - enabling IB Use Inline Data
```

Only in 7.14:
```
    2  Found /sys/memory_peers/amdkfd/version
    1  Kernel version #.#
    1  Skipping /proc/config.gz (zcat available, file not found)
    1  RCCL version : #.#.#-HEAD:#b#ab#
    1  NET/IB : AINIC RoCEv# optimizations enabled: CTS Inline Data: Enabled; CTS Offload: Enabled; IB Use Inline: enabled; GDR Flush: disabled
    1  Assigned GIN plugin GIN_IB_PROXY to comm
    1  Assigned RMA plugin GIN_IB_PROXY to comm
    1  initialized internal alternative rsmi functionality
    1  GFX# cheap fence is OFF
    1  AllGather |     #.#/   #.# |     #.#/   #.# |     #.#/   #.# |    #.#/  #.# |    #.#/   #.# |    #.#/ #.# |     #.#/   #.# |     #.#/   #.# |     #.#/
```

### all_reduce

Only in 10.1:
```
  205  AllReduce impl selected: algo DDA-IPC
   13  Taking kernel-based collective path
   13  Launch COLL kernel: gridDim.x=# blockDim.x=# (p#pnChannels=#, p#pnChannelsPerPeer=#, nChannels=#, nWorkBatches=#)
    2  Found peer memory client /sys/memory_peers/amdkfd
    1  Socket handshake magic #x# (built-in default)
    1  cuMem auto-enable is limited to gfx# (detected gfx#); set NCCL_CUMEM_ENABLE=# to override
    1  RCCL version : #.#.#-HEAD:#da#
    1  NET/IB : IB Use Inline is disabled and CTS Offload is enabled - enabling IB Use Inline Data
    1  NET/IB : AINIC RoCEv# optimizations enabled: CTS Inline Data: Enabled; CTS Offload: Enabled; IB Use Inline: Enabled; GDR Flush: Disabled
    1  ncclOsDlopen(libnccl-rma.so) failed: libnccl-rma.so: cannot open shared object file: No such file or directory
```

Only in 7.14:
```
    2  Found /sys/memory_peers/amdkfd/version
    1  Kernel version #.#
    1  Skipping /proc/config.gz (zcat available, file not found)
    1  RCCL version : #.#.#-HEAD:#b#ab#
    1  NET/IB : AINIC RoCEv# optimizations enabled: CTS Inline Data: Enabled; CTS Offload: Enabled; IB Use Inline: enabled; GDR Flush: disabled
    1  Assigned GIN plugin GIN_IB_PROXY to comm
    1  Assigned RMA plugin GIN_IB_PROXY to comm
    1  initialized internal alternative rsmi functionality
    1  GFX# cheap fence is OFF
    1  AllGather |     #.#/   #.# |     #.#/   #.# |     #.#/   #.# |    #.#/  #.# |    #.#/   #.# |    #.#/ #.# |     #.#/   #.# |     #.#/   #.# |     #.#/
```

### all_gather

Only in 10.1:
```
  169  AllGather impl selected: algo DDA-IPC
   49  RCCL DIRECT ALLGATHER disabled on AINIC.
    2  Found peer memory client /sys/memory_peers/amdkfd
    1  Socket handshake magic #x# (built-in default)
    1  cuMem auto-enable is limited to gfx# (detected gfx#); set NCCL_CUMEM_ENABLE=# to override
    1  RCCL version : #.#.#-HEAD:#da#
    1  NET/IB : IB Use Inline is disabled and CTS Offload is enabled - enabling IB Use Inline Data
    1  NET/IB : AINIC RoCEv# optimizations enabled: CTS Inline Data: Enabled; CTS Offload: Enabled; IB Use Inline: Enabled; GDR Flush: Disabled
    1  ncclOsDlopen(libnccl-rma.so) failed: libnccl-rma.so: cannot open shared object file: No such file or directory
    1  RMA/Plugin: Could not find: libnccl-rma.so
```

Only in 7.14:
```
    2  Found /sys/memory_peers/amdkfd/version
    1  Kernel version #.#
    1  Skipping /proc/config.gz (zcat available, file not found)
    1  RCCL version : #.#.#-HEAD:#b#ab#
    1  NET/IB : AINIC RoCEv# optimizations enabled: CTS Inline Data: Enabled; CTS Offload: Enabled; IB Use Inline: enabled; GDR Flush: disabled
    1  Assigned GIN plugin GIN_IB_PROXY to comm
    1  Assigned RMA plugin GIN_IB_PROXY to comm
    1  initialized internal alternative rsmi functionality
    1  GFX# cheap fence is OFF
    1  AllGather |     #.#/   #.# |     #.#/   #.# |     #.#/   #.# |    #.#/  #.# |    #.#/   #.# |    #.#/ #.# |     #.#/   #.# |     #.#/   #.# |     #.#/
```

### broadcast

Only in 10.1:
```
  218  Taking kernel-based collective path
  218  Launch COLL kernel: gridDim.x=# blockDim.x=# (p#pnChannels=#, p#pnChannelsPerPeer=#, nChannels=#, nWorkBatches=#)
    2  Found peer memory client /sys/memory_peers/amdkfd
    1  Socket handshake magic #x# (built-in default)
    1  cuMem auto-enable is limited to gfx# (detected gfx#); set NCCL_CUMEM_ENABLE=# to override
    1  RCCL version : #.#.#-HEAD:#da#
    1  NET/IB : IB Use Inline is disabled and CTS Offload is enabled - enabling IB Use Inline Data
    1  NET/IB : AINIC RoCEv# optimizations enabled: CTS Inline Data: Enabled; CTS Offload: Enabled; IB Use Inline: Enabled; GDR Flush: Disabled
    1  ncclOsDlopen(libnccl-rma.so) failed: libnccl-rma.so: cannot open shared object file: No such file or directory
    1  RMA/Plugin: Could not find: libnccl-rma.so
```

Only in 7.14:
```
    2  Found /sys/memory_peers/amdkfd/version
    1  Kernel version #.#
    1  Skipping /proc/config.gz (zcat available, file not found)
    1  RCCL version : #.#.#-HEAD:#b#ab#
    1  NET/IB : AINIC RoCEv# optimizations enabled: CTS Inline Data: Enabled; CTS Offload: Enabled; IB Use Inline: enabled; GDR Flush: disabled
    1  Assigned GIN plugin GIN_IB_PROXY to comm
    1  Assigned RMA plugin GIN_IB_PROXY to comm
    1  initialized internal alternative rsmi functionality
    1  GFX# cheap fence is OFF
    1  AllGather |     #.#/   #.# |     #.#/   #.# |     #.#/   #.# |    #.#/  #.# |    #.#/   #.# |    #.#/ #.# |     #.#/   #.# |     #.#/   #.# |     #.#/
```

### alltoall

Only in 10.1:
```
    2  Found peer memory client /sys/memory_peers/amdkfd
    1  Socket handshake magic #x# (built-in default)
    1  cuMem auto-enable is limited to gfx# (detected gfx#); set NCCL_CUMEM_ENABLE=# to override
    1  RCCL version : #.#.#-HEAD:#da#
    1  NET/IB : IB Use Inline is disabled and CTS Offload is enabled - enabling IB Use Inline Data
    1  NET/IB : AINIC RoCEv# optimizations enabled: CTS Inline Data: Enabled; CTS Offload: Enabled; IB Use Inline: Enabled; GDR Flush: Disabled
    1  ncclOsDlopen(libnccl-rma.so) failed: libnccl-rma.so: cannot open shared object file: No such file or directory
    1  RMA/Plugin: Could not find: libnccl-rma.so
    1  RMA/Plugin: Failed to load external plugin libnccl-rma.so, dlHandle: (nil), ncclRma: (nil)
    1  RMA/Plugin: Assigned plugin RMA_IB_PROXY to comm
```

Only in 7.14:
```
    2  Found /sys/memory_peers/amdkfd/version
    1  Kernel version #.#
    1  Skipping /proc/config.gz (zcat available, file not found)
    1  RCCL version : #.#.#-HEAD:#b#ab#
    1  NET/IB : AINIC RoCEv# optimizations enabled: CTS Inline Data: Enabled; CTS Offload: Enabled; IB Use Inline: enabled; GDR Flush: disabled
    1  Assigned GIN plugin GIN_IB_PROXY to comm
    1  Assigned RMA plugin GIN_IB_PROXY to comm
    1  initialized internal alternative rsmi functionality
    1  GFX# cheap fence is OFF
    1  AllGather |     #.#/   #.# |     #.#/   #.# |     #.#/   #.# |    #.#/  #.# |    #.#/   #.# |    #.#/ #.# |     #.#/   #.# |     #.#/   #.# |     #.#/
```

