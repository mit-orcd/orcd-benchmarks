#### 5.1.1 Why the 1-node numbers changed

**Main factors for small-message latency in one node:** (1) fixed cost per call (kernel launch, set-up, final sync), which dominates at 8 B-1 KB; (2) algorithm (ring, tree, direct), which sets how many steps and GPU-to-GPU handshakes a call needs; (3) protocol (LL, LL128, Simple); (4) number of channels; (5) implementation path (copy through RCCL buffers, or read peer GPU memory directly); (6) the tuning table that picks algorithm and protocol per size.

**Same in 7.14 and 10.1** (debug logs): channels (112 / 64 / 8), algorithm and protocol where logged (e.g. broadcast RING/LL up to 64K on both), and the tuning table (the new 10.1 rows are all_reduce at 32 MB and above).

**What differs and why:**
- **Implementation path (5):** RCCL 2.30.7 adds a new DDA-IPC path for all_reduce and all_gather. Each GPU reads the other 7 GPUs' buffers directly in one step, instead of passing data around a ring in 7 steps with a handshake each. Fewer steps, less waiting: this is the likely reason all_gather is about 30% faster and all_reduce 1M about 20% faster.
- **Fixed cost per call (1), probably:** sendrecv 1K and broadcast 1K are about 30% slower. Their algorithm, protocol and channels are the same, and the change is the same with DDA on or off, so it points to a higher per-call cost in RCCL 2.30.7 or HIP 7.16 (a deduction, not measured directly).
- **Not explained:** alltoall is about 30% faster without using DDA, so it comes from another part of the 2.30.7 rework; the logs do not say which.

**Notes:** this is not noise (repeat runs agree within about ±3%; the changes are 20-37%). `RCCL_DDA_ENABLE=0` does not restore 7.14: it falls back to a path slower than both versions (12 of 21 points more than 10% slower), so it cannot show how much DDA explains. Details, same-day timing and RCCL's choices: [`rccl-lat-1node-rocm10.1-why.md`](rccl-lat-1node-rocm10.1-why.md).

| Collective | Size | 7.14 (µs) | 10.1 (µs) | 10.1, DDA off (µs) |
|---|---:|---:|---:|---:|
| sendrecv | 1K | 6.5 | 8.4 | 8.5 |
| all_reduce | 1M | 17.4 | 13.7 | 35.1 |
| all_gather | 1K | 7.8 | 5.0 | 13.4 |
| all_gather | 64K | 9.0 | 6.3 | 14.7 |
| all_gather | 1M | 11.7 | 7.9 | 30.5 |
| broadcast | 1K | 5.8 | 7.7 | 8.0 |
| alltoall | 1K | 7.5 | 4.9 | 9.2 |
| alltoall | 64K | 10.4 | 7.6 | 11.2 |
| alltoall | 1M | 14.0 | 9.2 | 15.7 |
