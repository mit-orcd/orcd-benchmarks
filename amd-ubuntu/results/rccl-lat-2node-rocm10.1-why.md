# Why 2-node all_reduce latency differs between ROCm 7.14 and 10.1 (MI355X)

node6100 + node6101, 2 nodes × 8 MI355X, 8 RoCE rails (RCCL NET/IB). Run dir: `logs/node6100/rocm10.1/rccl/why_2node_20261030_170413`. Script: [`../rocm10.1/run_rccl_why_2node.sh`](../rocm10.1/run_rccl_why_2node.sh). Question from [`rocm10.1-vs-rocm7.14.md`](rocm10.1-vs-rocm7.14.md) §5.2: there, 7.14 runs RCCL's default and 10.1 runs `NCCL_PROTO=LL128` (10.1's default gives wrong results), so the all_reduce rows compare different settings. Here both versions run the same settings, back to back.

## Answer

- Both versions pick the same algorithm and protocol by default (8: TREE/LL; 1K: TREE/LL; 8K: TREE/LL; 64K: TREE/LL; 256K: TREE/LL; 1M: TREE/LL128).
- Default setting, data check: wrong results on 7.14 at no size, on 10.1 at 16K, 32K, 128K, 256K.
- Tree/LL forced on both: wrong results on 7.14 at no size, on 10.1 at 256K, 512K; 10.1 is more than 3× slower at 8, 1K, 8K, 64K, 256K (e.g. 8: 28.7 vs 1,426 µs).
- So the 2-node default problem on 10.1 is RCCL 2.30.7's Tree/LL path across nodes: 7.14 runs the same algorithm and protocol correctly and fast. This is a regression in RCCL, not a setting in this benchmark.
- Same setting on both (`NCCL_PROTO=LL128`), 10.1 vs 7.14: 8 -2.6%; 1K -2.1%; 8K -3.6%; 64K -3.2%; 256K -5.1%; 1M -17.4%. 1 sizes differ by more than 10% even on the same setting.
- On 7.14 itself, LL128 vs its default: 8 +43.7%; 1K +31.2%; 8K +37.3%; 64K +11.9%; 256K -0.6%; 1M +5.3% (positive = LL128 slower).
- 10.1 with `RCCL_DDA_ENABLE=0`: wrong results at 64K; 8 B 1,416 µs vs 1,416 µs default, so DDA is not behind the 2-node problem.

## 1. Latency (time per call, µs, out-of-place, lower is better)

rccl-tests all_reduce, 1000 iterations + 100 warm-up, no data check, NUMA-bound ranks; one run per column. Change = (10.1 − 7.14) / 7.14, so a negative number means 10.1 is faster.

| Size | default, 7.14 | default, 10.1 | Change | `NCCL_PROTO=LL128`, 7.14 | `NCCL_PROTO=LL128`, 10.1 | Change | `NCCL_ALGO=Tree NCCL_PROTO=LL`, 7.14 | `NCCL_ALGO=Tree NCCL_PROTO=LL`, 10.1 | Change | `NCCL_ALGO=Ring NCCL_PROTO=LL`, 7.14 | `NCCL_ALGO=Ring NCCL_PROTO=LL`, 10.1 | Change | `RCCL_DDA_ENABLE=0`, 10.1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 28.0 | 1,416 | +4950.6% | 40.3 | 39.2 | -2.6% | 28.7 | 1,426 | +4872.8% | 53.7 | 53.5 | -0.4% | 1,416 |
| 1K | 30.5 | 1,694 | +5449.8% | 40.0 | 39.2 | -2.1% | 31.4 | 1,695 | +5298.4% | 64.0 | 64.6 | +0.9% | 1,684 |
| 8K | 33.8 | 2,160 | +6295.0% | 46.4 | 44.7 | -3.6% | 33.9 | 2,158 | +6265.4% | 64.4 | 65.0 | +0.8% | 2,160 |
| 64K | 41.9 | 2,168 | +5070.0% | 47.0 | 45.4 | -3.2% | 41.5 | 2,169 | +5131.3% | 68.1 | 69.0 | +1.3% | 2,169 |
| 256K | 55.5 | 2,179 | +3829.8% | 55.1 | 52.3 | -5.1% | 55.3 | 2,179 | +3837.1% | 77.0 | 77.5 | +0.7% | 2,179 |
| 1M | 66.2 | 57.2 | -13.6% | 69.7 | 57.6 | -17.4% | 86.0 | 2,196 | +2452.0% | 127 | 123 | -2.8% | 57.3 |

## 2. Correctness (data check on, 20 iterations, 8 B to 1 MB)

| Setting | 7.14: sizes with wrong results | 10.1: sizes with wrong results |
|---|---|---|
| default | none | 16K, 32K, 128K, 256K |
| `NCCL_PROTO=LL128` | none | none |
| `NCCL_ALGO=Tree NCCL_PROTO=LL` | none | 256K, 512K |
| `NCCL_ALGO=Ring NCCL_PROTO=LL` | none | none |
| `RCCL_DDA_ENABLE=0` | not run | 64K |

## 3. What RCCL picks by default (debug run, rank 0)

`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL`, 2 iterations.

| Size | 7.14 | 10.1 |
|---:|---|---|
| 8 | TREE/LL | TREE/LL |
| 1K | TREE/LL | TREE/LL |
| 8K | TREE/LL | TREE/LL |
| 64K | TREE/LL | TREE/LL |
| 256K | TREE/LL | TREE/LL |
| 1M | TREE/LL128 | TREE/LL128 |

