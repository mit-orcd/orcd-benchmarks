# Megatron-LM 1-node GPU sweep — B200

- Generated: 2026-08-06 20:00:52
- Nodes: node5500, node5501, node5502 (single node each, data-parallel, TP=1, PP=1)
- Model: ~7B GPT — 36 layers, hidden 4096, FFN 14336, 32 heads, seq 2048, bf16
- Per run: micro-batch 4, global batch = 128 x total_GPUs, 100 iters, no activation recompute
- Metric: last-iteration throughput (TFLOP/s/GPU), same as the reference
- Reference: MIT aicr-benchmarks `megatron-lm/output/summary.md`, B200 1-node group

## Apples-to-apple vs B200 reference

| #GPUs | GBS | reference TFLOP/s/GPU | node5500 TFLOP/s/GPU | node5500 / ref | node5501 TFLOP/s/GPU | node5501 / ref | node5502 TFLOP/s/GPU | node5502 / ref |
|------:|----:|----------------------:|-----------------:|-----------:|-----------------:|-----------:|-----------------:|-----------:|
| 1 | 128 | 1024.4 | 989.0 | 96.5% | — | — | — | — |
| 2 | 256 | 1007.7 | 982.9 | 97.5% | — | — | — | — |
| 4 | 512 | 985.2 | 967.0 | 98.2% | — | — | — | — |
| 8 | 1024 | 993.3 | 967.4 | 97.4% | 963.9 | 97.0% | 975.1 | 98.2% |

Reference values are the best B200 1-node result per GPU count from `summary.md` (last-iteration TFLOP/s/GPU).

## Scaling (1 -> 8 GPUs, single node)

### node5500

| #GPUs | GBS | per-GPU TFLOP/s | aggregate TFLOP/s | iter (ms) | weak-scaling eff. | status |
|------:|----:|----------------:|------------------:|----------:|------------------:|--------|
| 1 | 128 | 989.0 | 989 | 11374 | 100.0% | ok |
| 2 | 256 | 982.9 | 1966 | 11446 | 99.4% | ok |
| 3 | 384 | 969.8 | 2909 | 11599 | 98.1% | ok |
| 4 | 512 | 967.0 | 3868 | 11634 | 97.8% | ok |
| 5 | 640 | 966.7 | 4834 | 11637 | 97.7% | ok |
| 6 | 768 | 969.1 | 5815 | 11608 | 98.0% | ok |
| 7 | 896 | 968.6 | 6780 | 11615 | 97.9% | ok |
| 8 | 1024 | 967.4 | 7739 | 11629 | 97.8% | ok |

### node5501

| #GPUs | GBS | per-GPU TFLOP/s | aggregate TFLOP/s | iter (ms) | weak-scaling eff. | status |
|------:|----:|----------------:|------------------:|----------:|------------------:|--------|
| 8 | 1024 | 963.9 | 7711 | 11671 | — | ok |

### node5502

| #GPUs | GBS | per-GPU TFLOP/s | aggregate TFLOP/s | iter (ms) | weak-scaling eff. | status |
|------:|----:|----------------:|------------------:|----------:|------------------:|--------|
| 8 | 1024 | 975.1 | 7801 | 11537 | — | ok |

Aggregate = per-GPU x #GPUs. Weak-scaling efficiency = per-GPU(N) / per-GPU(1) on that node. Per-GPU work is held constant (GBS scales with #GPUs).

## Multi-node runs

| Node set | nodes | GPUs/node | total GPUs | GBS | per-GPU TFLOP/s | aggregate TFLOP/s | iter (ms) | vs best 1-node per-GPU | status |
|----------|------:|----------:|-----------:|----:|----------------:|------------------:|----------:|----------------------:|--------|
| 5500-5501 | 2 | 8 | 16 | 2048 | 962.0 | 15392 | 11694 | 98.7% | ok |
| 5500-5502 | 2 | 8 | 16 | 2048 | 966.8 | 15469 | 11636 | 99.1% | ok |
| 5501-5502 | 2 | 8 | 16 | 2048 | 957.6 | 15322 | 11747 | 98.2% | ok |
| 5500-5501-5502 | 3 | 8 | 24 | 3072 | 956.6 | 22958 | 11760 | 98.1% | ok |

Multi-node runs cross the InfiniBand fabric for gradient all-reduce instead of staying on NVLink, so per-GPU throughput below the 1-node rate is the cost of inter-node gradient sync. Weak scaling is preserved: GBS = 128 x total GPUs, so per-GPU work (and the 32 gradient-accumulation steps) is identical across every run. `vs best 1-node per-GPU` compares against the best single-node result at the same GPUs-per-node.

## Scaling figure

![Aggregate TFLOP/s vs number of GPUs](megatron-scaling.svg)

Aggregate throughput vs #GPUs: one curve per node, ideal linear scaling from the best 1-GPU point (dashed), and the B200 reference (orange).

