# gpu-fryer summary

- Generated: 2026-07-13 16:24:13
- Nodes: node5500, node5502 (8 x NVIDIA B200 each)
- Precisions: FP32, BF16, FP8
- Reference (MIT aicr-benchmarks, `gpu-fryer/summary.md`, b0025): per-GPU mean TFLOP/s — FP32 772, BF16 1500, FP8 4115

## Per-node mean converged throughput (TFLOP/s)

| Node | FP32 | BF16 | FP8 | Health |
|------|------:|------:|------:|---|
| node5500 | 746 | 1454 | 3989 | ok |
| node5502 | 751 | 1461 | 4011 | ok |
| **reference (b0025)** | **772** | **1500** | **4115** | — |

### % of B200 reference (mean)

| Node | FP32 | BF16 | FP8 |
|------|------:|------:|------:|
| node5500 | 97% | 97% | 97% |
| node5502 | 97% | 97% | 97% |

## Per-GPU converged throughput (TFLOP/s)

### node5500

| GPU | FP32 | BF16 | FP8 |
|-----|------:|------:|------:|
| 0 | 742.1 | 1476.0 | 4044.7 |
| 1 | 756.6 | 1470.8 | 4059.5 |
| 2 | 740.1 | 1437.4 | 3928.2 |
| 3 | 740.5 | 1437.4 | 3940.0 |
| 4 | 757.1 | 1466.0 | 4006.7 |
| 5 | 749.9 | 1456.6 | 4030.8 |
| 6 | 745.1 | 1446.8 | 3920.7 |
| 7 | 740.6 | 1442.1 | 3978.9 |
| **min** | **740.1** | **1437.4** | **3920.7** |
| **mean** | **746.5** | **1454.1** | **3988.7** |
| **max** | **757.1** | **1476.0** | **4059.5** |

### node5502

| GPU | FP32 | BF16 | FP8 |
|-----|------:|------:|------:|
| 0 | 752.1 | 1468.6 | 4033.6 |
| 1 | 759.3 | 1467.5 | 4069.7 |
| 2 | 746.2 | 1460.9 | 3987.1 |
| 3 | 758.0 | 1470.4 | 4040.0 |
| 4 | 755.9 | 1471.1 | 4022.1 |
| 5 | 746.7 | 1453.7 | 3955.5 |
| 6 | 744.9 | 1448.1 | 4003.8 |
| 7 | 744.4 | 1449.5 | 3980.1 |
| **min** | **744.4** | **1448.1** | **3955.5** |
| **mean** | **751.0** | **1461.2** | **4011.5** |
| **max** | **759.3** | **1471.1** | **4069.7** |

Converged = the final sustained-average throughput gpu-fryer reports per GPU at the end of each precision run. Higher is better; large spread across GPUs or any throttling flag indicates a problem.

