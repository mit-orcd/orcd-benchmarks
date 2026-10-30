#### 5.2.1 Why 2-node all_reduce differs

- Both versions pick the same algorithm and protocol by default (8: TREE/LL; 1K: TREE/LL; 8K: TREE/LL; 64K: TREE/LL; 256K: TREE/LL; 1M: TREE/LL128).
- Default setting, data check: wrong results on 7.14 at no size, on 10.1 at 16K, 32K, 128K, 256K.
- Tree/LL forced on both: wrong results on 7.14 at no size, on 10.1 at 256K, 512K; 10.1 is more than 3× slower at 8, 1K, 8K, 64K, 256K (e.g. 8: 28.7 vs 1,426 µs).
- So the 2-node default problem on 10.1 is RCCL 2.30.7's Tree/LL path across nodes: 7.14 runs the same algorithm and protocol correctly and fast. This is a regression in RCCL, not a setting in this benchmark.
- Same setting on both (`NCCL_PROTO=LL128`), 10.1 vs 7.14: 8 -2.6%; 1K -2.1%; 8K -3.6%; 64K -3.2%; 256K -5.1%; 1M -17.4%. 1 sizes differ by more than 10% even on the same setting.
- On 7.14 itself, LL128 vs its default: 8 +43.7%; 1K +31.2%; 8K +37.3%; 64K +11.9%; 256K -0.6%; 1M +5.3% (positive = LL128 slower).
- 10.1 with `RCCL_DDA_ENABLE=0`: wrong results at 64K; 8 B 1,416 µs vs 1,416 µs default, so DDA is not behind the 2-node problem.

Same-day runs with the same settings on both versions, correctness checks and RCCL's choices: [`rccl-lat-2node-rocm10.1-why.md`](rccl-lat-2node-rocm10.1-why.md).
