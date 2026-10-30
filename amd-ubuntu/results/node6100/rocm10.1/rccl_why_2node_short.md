#### 5.2.1 Why the 2-node numbers changed

**Main factors for small-message latency across 2 nodes:** the same as in one node (fixed cost per call, algorithm, protocol, channels, tuning table), plus the network: the NIC and the switch hop between nodes (8 RoCE rails), and RCCL's CPU proxy thread that drives the network sends. The network part adds about 10-30 µs per call, so it dominates at small sizes.

**Same in 7.14 and 10.1:** the hardware, the network and the default algorithm and protocol (all_reduce: Tree/LL up to 256K, Tree/LL128 at 1M).

**What differs and why:**
- **all_reduce, protocol (8 B-64K slower on 10.1):** RCCL 2.30.7's default Tree/LL path across nodes is broken (wrong results, 1.4-2.2 ms per call), a regression in RCCL. So the 10.1 rows use `NCCL_PROTO=LL128`. LL128 has more overhead per call than LL at small sizes: on 7.14 itself LL128 is 31-44% slower than LL at 8 B-8K. With the same setting (LL128) on both, 10.1 is within about 5% of 7.14 up to 256K. So the slow-down comes from the protocol forced by the bug, not from a slower RCCL.
- **all_reduce 1M (faster on 10.1):** both use Tree/LL128 here, and 10.1 is 14-17% faster; the logs do not show why.
- **sendrecv small (6-10% faster on 10.1), possible cause:** RCCL 2.30.7 turns on InfiniBand inline data (its log says "enabling IB Use Inline Data"), which sends small messages with less work on the NIC. Not tested directly.
- **Everything else** (all_gather, reduce_scatter, broadcast, alltoall, and 5.3 with 1 GPU per node) is within about 6%: these use the same network path, and the DDA change in one node does not apply across nodes.

**Details (all_reduce checks):**
- Both versions pick the same algorithm and protocol by default (8: TREE/LL; 1K: TREE/LL; 8K: TREE/LL; 64K: TREE/LL; 256K: TREE/LL; 1M: TREE/LL128).
- Default setting, data check: wrong results on 7.14 at no size, on 10.1 at 16K, 32K, 128K, 256K.
- Tree/LL forced on both: wrong results on 7.14 at no size, on 10.1 at 256K, 512K; 10.1 is more than 3× slower at 8, 1K, 8K, 64K, 256K (e.g. 8: 28.7 vs 1,426 µs).
- So the 2-node default problem on 10.1 is RCCL 2.30.7's Tree/LL path across nodes: 7.14 runs the same algorithm and protocol correctly and fast. This is a regression in RCCL, not a setting in this benchmark.
- Same setting on both (`NCCL_PROTO=LL128`), 10.1 vs 7.14: 8 -2.6%; 1K -2.1%; 8K -3.6%; 64K -3.2%; 256K -5.1%; 1M -17.4%. 1 sizes differ by more than 10% even on the same setting.
- On 7.14 itself, LL128 vs its default: 8 +43.7%; 1K +31.2%; 8K +37.3%; 64K +11.9%; 256K -0.6%; 1M +5.3% (positive = LL128 slower).
- 10.1 with `RCCL_DDA_ENABLE=0`: wrong results at 64K; 8 B 1,416 µs vs 1,416 µs default, so DDA is not behind the 2-node problem.

Same-day runs with the same settings on both versions, correctness checks and RCCL's choices: [`rccl-lat-2node-rocm10.1-why.md`](rccl-lat-2node-rocm10.1-why.md).
