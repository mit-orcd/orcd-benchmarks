# Megatron-LM 1-node GPU sweep — B200

- Generated: 2026-10-05 14:22:02
- Nodes: node5900-c1 (single node each, data-parallel, TP=1, PP=1)
- Model: ~7B GPT — 36 layers, hidden 4096, FFN 14336, 32 heads, seq 2048, bf16
- Per run: micro-batch 4, global batch = 128 x total_GPUs, 100 iters, no activation recompute
- Metric: last-iteration throughput (TFLOP/s/GPU), same as the reference
- Reference: MIT aicr-benchmarks `megatron-lm/output/summary.md`, B200 1-node group

## Apples-to-apple vs B200 reference

| #GPUs | GBS | reference TFLOP/s/GPU | node5900-c1 TFLOP/s/GPU | node5900-c1 / ref |
|------:|----:|----------------------:|-----------------:|-----------:|
| 1 | 128 | 1024.4 | 1021.2 | 99.7% |
| 2 | 256 | 1007.7 | 1030.8 | 102.3% |
| 4 | 512 | 985.2 | 1018.3 | 103.4% |
| 8 | 1024 | 993.3 | 1003.2 | 101.0% |

Reference values are the best B200 1-node result per GPU count from `summary.md` (last-iteration TFLOP/s/GPU).

## Scaling (1 -> 8 GPUs, single node)

### node5900-c1

| #GPUs | GBS | per-GPU TFLOP/s | aggregate TFLOP/s | iter (ms) | weak-scaling eff. | status |
|------:|----:|----------------:|------------------:|----------:|------------------:|--------|
| 1 | 128 | 1021.2 | 1021 | 11016 | 100.0% | ok |
| 2 | 256 | 1030.8 | 2062 | 10913 | 100.9% | ok |
| 3 | 384 | 1025.3 | 3076 | 10972 | 100.4% | ok |
| 4 | 512 | 1018.3 | 4073 | 11048 | 99.7% | ok |
| 5 | 640 | 1015.0 | 5075 | 11084 | 99.4% | ok |
| 6 | 768 | 1014.2 | 6085 | 11092 | 99.3% | ok |
| 7 | 896 | 1018.3 | 7128 | 11047 | 99.7% | ok |
| 8 | 1024 | 1003.2 | 8026 | 11214 | 98.2% | ok |

Aggregate = per-GPU x #GPUs. Weak-scaling efficiency = per-GPU(N) / per-GPU(1) on that node. Per-GPU work is held constant (GBS scales with #GPUs).

## Scaling figure

![Aggregate TFLOP/s vs number of GPUs](megatron-scaling.svg)

Aggregate throughput vs #GPUs: one curve per node, ideal linear scaling from the best 1-GPU point (dashed), and the B200 reference (orange).

