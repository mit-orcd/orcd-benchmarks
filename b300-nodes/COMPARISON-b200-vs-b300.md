# B200 vs B300 — single-node benchmark comparison

- Generated: 2026-10-01 13:00:30
- B300 node: node5900-c1 (mit_testing, 8 x B300); B200 data from `../b200-nodes/`
- Ratios are B300 / B200; > 1.00x means B300 is faster.
- Per-benchmark B300 summaries: `out-gpu-fryer/summary.md`, `out-nccl-1node/summary.md`, `output-megatron/summary.md`

## 1. gpu-fryer — per-GPU matmul throughput (TFLOP/s)

B200 = mean over 10 B200 node(s) of the per-node mean (newest run per node); B300 = node5900-c1 (B300 SXM6 AC). Converged value per GPU, 300 s per precision.

| Precision | B200 mean | B200 node range | B300 mean | B300 GPU range | B300 / B200 |
|---|---:|---:|---:|---:|---:|
| FP32 | 749 | 740–760 | 797 | 781–812 | **1.06x** |
| BF16 | 1,456 | 1,437–1,465 | 1,521 | 1,495–1,546 | **1.04x** |
| FP8 | 4,008 | 3,949–4,062 | 4,228 | 4,226–4,233 | **1.05x** |

Throttling reported on B300: none

### B300 per-GPU detail — node5900-c1

| GPU | FP32 | BF16 | FP8 |
|---|---:|---:|---:|
| GPU0 | 792 | 1,513 | 4,226 |
| GPU1 | 800 | 1,528 | 4,226 |
| GPU2 | 800 | 1,529 | 4,226 |
| GPU3 | 798 | 1,524 | 4,226 |
| GPU4 | 799 | 1,526 | 4,233 |
| GPU5 | 812 | 1,546 | 4,233 |
| GPU6 | 794 | 1,507 | 4,226 |
| GPU7 | 781 | 1,495 | 4,226 |

## 2. NCCL 1-node — intra-node NVLink bus bandwidth (GB/s)

Converged busbw = busbw at the largest message (16 GiB), best of out-of-place / in-place. B200 = mean over the B200 nodes that ran that collective (newest run per node); 8 GPUs, 1 MPI task.

### node5900-c1 (8 x NVIDIA B300 SXM6 AC)

| Collective | B200 busbw | #B200 nodes | B300 busbw | B300 peak | B300 / B200 | B300 correctness |
|---|---:|---:|---:|---:|---:|---:|
| sendrecv | 668.6 | 10 | 638.4 | 638.4 | **0.95x** | ok |
| reduce | 687.5 | 10 | 657.3 | 677.1 | **0.96x** | ok |
| broadcast | 682.1 | 10 | 670.5 | 713.1 | **0.98x** | ok |
| gather | 718.0 | 10 | 797.5 | 797.5 | **1.11x** | ok |
| scatter | 731.2 | 10 | 689.5 | 704.8 | **0.94x** | ok |
| reduce_scatter | 691.6 | 10 | 759.9 | 759.9 | **1.10x** | ok |
| all_gather | 681.6 | 10 | 646.0 | 646.0 | **0.95x** | ok |
| all_reduce | 839.4 | 10 | 914.3 | 914.3 | **1.09x** | ok |
| alltoall | 658.1 | 10 | 628.0 | 628.0 | **0.95x** | ok |

#### all_reduce busbw vs message size (out-of-place)

| Size | B200 | B300 | B300 / B200 |
|---|---:|---:|---:|
| 1 MiB | 34.4 | 38.7 | **1.13x** |
| 4 MiB | 125.1 | 121.6 | **0.97x** |
| 16 MiB | 264.2 | 256.7 | **0.97x** |
| 64 MiB | 416.7 | 402.8 | **0.97x** |
| 256 MiB | 651.2 | 717.0 | **1.10x** |
| 1 GiB | 727.0 | 798.4 | **1.10x** |
| 4 GiB | 822.5 | 904.4 | **1.10x** |
| 16 GiB | 833.7 | 913.5 | **1.10x** |

## 3. ib_write_bw — GPUDirect RDMA, two rails of one node (Gb/s)

64 MiB RDMA write, 200 iterations. B200: `../b200-nodes/out-ibwrite/ibwrite-1node-20306762.out` (client mlx5_4, server mlx5_7). B300: `out-ibwrite/ibwrite-1node-24550230.out` (client mlx5_0, server mlx5_6).

| Test | B200 | B300 | B300 / B200 |
|---|---:|---:|---:|
| host mem -> host mem | 379.72 | 404.94 | **1.07x** |
| NIC reads from GPU | 395.48 | 373.56 | **0.94x** |
| NIC writes into GPU | 380.45 | 373.70 | **0.98x** |
| GPU -> GPU | 395.47 | 403.46 | **1.02x** |

### Size sweep, NIC reads from GPU

| Size | B200 | B300 | B300 / B200 |
|---|---:|---:|---:|
| 4 KiB | 174.89 | 162.81 | **0.93x** |
| 8 KiB | 289.88 | 324.00 | **1.12x** |
| 16 KiB | 365.22 | 398.86 | **1.09x** |
| 32 KiB | 355.93 | 401.21 | **1.13x** |
| 64 KiB | 357.17 | 402.64 | **1.13x** |
| 128 KiB | 376.76 | 403.15 | **1.07x** |
| 256 KiB | 378.05 | 403.56 | **1.07x** |
| 512 KiB | 386.41 | 403.72 | **1.04x** |
| 1 MiB | 390.20 | 403.75 | **1.03x** |
| 2 MiB | 394.12 | 403.82 | **1.02x** |
| 4 MiB | 394.84 | 403.84 | **1.02x** |
| 8 MiB | 395.08 | 403.84 | **1.02x** |

## 4. Megatron-LM 1-node — reference ~7B GPT (TFLOP/s/GPU)

Same config on both: 36 layers, hidden 4096, FFN 14336, seq 2048, bf16, micro-batch 4, global batch = 128 x GPUs, 100 iters, no recompute, TP=PP=1. Metric = last-iteration throughput per GPU. B200 = mean over B200 nodes with that GPU count (newest run).

| #GPUs | GBS | B200 TFLOP/s/GPU | #B200 nodes | B300 TFLOP/s/GPU | B300 aggregate | B200 iter ms | B300 iter ms | B300 / B200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 989.0 | 1 | — | — | 11,374 | — | — |
| 2 | 256 | 982.9 | 1 | — | — | 11,446 | — | — |
| 3 | 384 | 969.8 | 1 | — | — | 11,599 | — | — |
| 4 | 512 | 967.0 | 1 | 1,026.8 | 4,107 | 11,634 | 10,956 | **1.06x** |
| 5 | 640 | 966.7 | 1 | — | — | 11,637 | — | — |
| 6 | 768 | 969.1 | 1 | — | — | 11,608 | — | — |
| 7 | 896 | 968.6 | 1 | — | — | 11,615 | — | — |
| 8 | 1024 | 968.8 | 3 | — | — | 11,612 | — | — |

## 5. Megatron-LM 1-node — tuned max throughput, best per GPU type (TFLOP/s/GPU)

Each GPU type gets its own grid sweep (8 GPUs, seq 4096, distributed optimizer, overlapped grad-reduce/param-gather, TP=PP=1, grad-acc 4, 20 iters): 5B (24L, h4096) micro 4/8/16 x recompute none/selective/full, and 13B (40L, h5120) micro 2/4/8 x recompute none/full, each in bf16 and fp8. The configs are NOT forced to match: the best point per GPU type and precision is compared. TFLOP/s/GPU is Megatron's model-FLOP throughput (recompute work not counted). Peak memory = max allocated by PyTorch on rank 0.

| Precision | B200 best | B200 config | B200 mem GiB | B300 best | B300 config | B300 mem GiB | B300 / B200 |
|---|---:|---|---:|---:|---|---:|---:|
| bf16 | — | — | — | — | — | — | — |
| fp8 | — | — | — | — | — | — | — |

Grid points finished OK: B200 0/0, B300 0/1 (of 30 each).

### Full sweep grid (TFLOP/s/GPU, peak GiB)

| Prec | Model | Micro | Recompute | B200 | B300 | B300 / B200 |
|---|---|---:|---|---:|---:|---:|
| bf16 | 5b | 4 | none | — | failed/running | — |

