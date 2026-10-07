# B200 vs B300 — single-node benchmark comparison

- Generated: 2026-10-05 16:34:41
- B300 node: node5900-c1 (mit_testing, 8 x B300); B200 data from `../b200-nodes/`
- Ratios are B300 / B200; > 1.00x means B300 is faster.
- Per-benchmark B300 summaries: `out-gpu-fryer/summary.md`, `out-nccl-1node/summary.md`, `output-megatron/summary.md`

## 1. On paper — official NVIDIA specs, B300 vs B200

Sources: [NVIDIA HGX platform page](https://www.nvidia.com/en-us/data-center/hgx/) (HGX B300 vs HGX B200 table), [Inside NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) (NVIDIA technical blog), [DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/) and [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) product pages. Per-GPU values are the HGX 8-GPU numbers divided by 8; NVIDIA lists tensor-core numbers with 2:4 sparsity, and dense = 1/2 sparse except where NVIDIA gives a dense number (FP4).

| Per GPU (HGX, dense) | B200 (Blackwell) | B300 (Blackwell Ultra) | B300 / B200 |
|---|---|---|---:|
| FP4 tensor (NVFP4) | 9 PFLOP/s | 13.5 PFLOP/s | **1.50x** |
| FP8 / FP6 tensor | 4.5 PFLOP/s | 4.5 PFLOP/s | 1.00x |
| BF16 / FP16 tensor | 2.25 PFLOP/s | 2.25 PFLOP/s | 1.00x |
| TF32 tensor | 1.125 PFLOP/s | 1.125 PFLOP/s | 1.00x |
| FP32 (non-tensor) | 75 TFLOP/s | 75 TFLOP/s | 1.00x |
| INT8 tensor | 4.5 POP/s | ~0.19 POP/s | **~0.04x** |
| FP64 / FP64 tensor | 37 TFLOP/s | 1.25 TFLOP/s | **~0.03x** |
| Attention softmax (SFU exp) | 5 T exp/s | 10.7 T exp/s | **2.14x** |
| HBM3E capacity | 180 GB (8-high stacks) | 288 GB (8 x 12-high stacks; ~270 GB usable on HGX) | **1.5x** |
| HBM bandwidth | 8 TB/s | 8 TB/s | 1.00x |
| NVLink 5 GPU-to-GPU | 1.8 TB/s | 1.8 TB/s | 1.00x |
| Scale-out NIC per GPU | ConnectX-7, 400 Gb/s | ConnectX-8, 800 Gb/s (PCIe Gen6) | **2.0x** |
| Max GPU power (HGX) | 1,000 W | 1,100 W (up to 1,400 W in GB300 systems) | 1.10x |
| Process / transistors | TSMC 4NP, 208 B | TSMC 4NP, 208 B (160 SMs) | same |

B300 advantages on paper:

- **1.5x memory per GPU** (288 GB vs 180-192 GB). This allows larger models or longer contexts per GPU, bigger micro-batches, less activation recompute, and bigger KV caches for inference.
- **1.5x dense FP4 (NVFP4)**, useful for FP4 inference and FP4 training recipes.
- **~2x attention-layer exponent throughput**, which speeds up softmax-heavy attention, mainly at long context.
- **2x scale-out network per GPU** (ConnectX-8 800 Gb/s vs ConnectX-7 400 Gb/s), which helps multi-node training and inference.
- Higher power limit (1,100 W vs 1,000 W per GPU), so clocks may hold up better under sustained load.

B300 disadvantages on paper:

- **FP64 drops ~30x** (37 TFLOP/s to 1.25 TFLOP/s per GPU). B300 is a poor fit for double-precision HPC codes (CFD, MD with FP64, dense linear algebra, HPL).
- **INT8 tensor drops ~24x**. Legacy INT8 inference paths should move to FP8/FP4.
- **No gain for FP8, BF16, TF32 or FP32 dense math, HBM bandwidth, or NVLink.** Standard BF16/FP8 training throughput is expected to be similar to B200. Any gain comes from extra memory (bigger batches, no recompute), faster attention, and power headroom.
- More power and heat per GPU (+10% on HGX), plus a newer stack: the CUDA 12.8+/13 toolchain, compute capability 10.3, and a newer driver.

What this predicts for the measurements below: gpu-fryer BF16/FP8 and NCCL NVLink close to 1.0x; ib_write_bw up to ~2x if the B300 links run at 800 Gb/s; Megatron reference config ~1.0x; tuned Megatron somewhat above B200 thanks to the larger memory.

## 2. gpu-fryer — per-GPU matmul throughput (TFLOP/s)

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

## 3. cuBLASLt GEMM — per-GPU throughput incl. FP4 (TFLOP/s)

From `../cuBLASLt/summary.md` (separate benchmark: `../cuBLASLt/job-lt-bench.sh`, 8 GPUs at once, tuned shape per GPU type; full method and per-shape results there).
- Generated: 2026-10-01 18:45:10
- Nodes: node5802-c1 (NVIDIA B200), node5900-c1 (NVIDIA B300 SXM6 AC)
- Library: cuBLASLt 13.8.0 (CUDA 13.4 redist, `install.sh`); program `src/lt-bench.c`

### Sustained throughput per GPU (TFLOP/s)

Sustained = tuned shape for 60 s, first 10 s dropped, what a long job gets.

| Precision | B200 | B300 | B300 / B200 |
|---|---:|---:|---:|
| FP4 (NVFP4) | 5,708 | 6,877 | **1.20x** |
| FP8 (E4M3) | 2,524 | 2,524 | **1.00x** |
| BF16 | 1,396 | 1,437 | **1.03x** |
| FP16 | 1,322 | 1,333 | **1.01x** |
| TF32 | 720 | 756 | **1.05x** |
| FP64 | 36 | 1.10 | **0.03x** |
| INT8 (TOPS) | 2,931 | 151 | **0.05x** |

Peak (tuned shape, 3 s run after the warm-up) is very close to sustained: within 0.5% for every precision on both GPUs.

### Clocks and power during the sustained run

Mean over the GPUs of the sustained run (nvidia-smi, 1 s samples, GPU utilization >= 90%).

| Precision | B200 SM MHz | B200 W | B200 max °C | B300 SM MHz | B300 W | B300 max °C |
|---|---:|---:|---:|---:|---:|---:|
| FP4 (NVFP4) | 1,314 | 993 | 73 | 1,099 | 1,091 | 74 |
| FP8 (E4M3) | 1,131 | 994 | 74 | 1,114 | 1,086 | 75 |
| BF16 | 1,299 | 991 | 74 | 1,241 | 1,091 | 76 |
| FP16 | 1,193 | 994 | 74 | 1,172 | 1,088 | 76 |
| TF32 | 1,360 | 990 | 74 | 1,402 | 1,089 | 75 |
| FP64 | 1,965 | 751 | 65 | 2,032 | 276 | 42 |
| INT8 (TOPS) | 1,291 | 991 | 73 | 2,032 | 371 | 47 |

### Against the datasheet (dense, approximate)

NVIDIA HGX B200 / B300 dense figures per GPU (half the "with sparsity" numbers); the B300 INT8 figure is not listed here. % = sustained / datasheet.

| Precision | B200 datasheet | B200 sustained % | B300 datasheet | B300 sustained % |
|---|---:|---:|---:|---:|
| FP4 (NVFP4) | 9,000 | 63% | 13,500 | 51% |
| FP8 (E4M3) | 4,500 | 56% | 4,500 | 56% |
| BF16 | 2,250 | 62% | 2,250 | 64% |
| FP16 | 2,250 | 59% | 2,250 | 59% |
| TF32 | 1,100 | 65% | 1,100 | 69% |
| FP64 | 37 | 97% | 1.25 | 88% |
| INT8 (TOPS) | 4,500 | 65% | - | - |

### Why FP4 is 1.20x measured but 1.5x on paper

(Explanation written from the 2026-10-01 cuBLASLt run; the numbers below are from the tables above.)

The gap is mainly the power limit. Both GPUs hit their power cap during the FP4 test, and B300 then has to run at a much lower clock. NVIDIA's spec-sheet numbers assume full clocks.

1. **Both GPUs are power-capped.** In the sustained FP4 run B200 drew 993 W against its 1,000 W limit and B300 drew 1,091 W against its 1,100 W limit. At the cap the GPU lowers its clock.
2. **B300 loses more clock than B200.** The 1.5x gain comes from 1.5x more FP4 math per clock cycle, which also costs more energy per cycle. B300 has only 10% more power than B200, so it slows down further.

| FP4, sustained | B200 | B300 |
|---|---:|---:|
| SM clock | 1,314 MHz | 1,099 MHz (0.84x of B200) |
| Share of the unthrottled clock (~1,965 / ~2,032 MHz, from the FP64 run) | 67% | 54% |
| Share of spec-sheet FP4 peak | 63% | 51% |
| FP4 work per watt (TFLOP/s per W) | 5.75 | 6.30 (1.10x) |

3. **The numbers add up.** 1.5x per clock x 0.84x clock = ~1.25x, close to the measured 1.20x. Both GPUs reach about 94% of what their actual clock allows (63/67 and 51/54), so the GEMM kernels are not the limit. Equivalently: 1.10x more work per watt x 1.10x more power = ~1.2x.
4. **Spec-sheet context.** NVIDIA's 15 PFLOP/s dense FP4 figure for Blackwell Ultra is for parts running at up to 1,400 W (GB300-class systems), not this 1,100 W HGX board.
5. **Other precisions.** FP4 is the only precision where B300 has more peak compute per clock. In BF16, FP8 and TF32 both GPUs have the same peak and run at similar clocks, so they measure about 1.0x.
6. **What would change it.** A higher GPU power limit should narrow the gap; that is set by the node administrators, not by user jobs.

## 4. NCCL 1-node — intra-node NVLink bus bandwidth (GB/s)

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

## 5. ib_write_bw — GPUDirect RDMA, two rails of one node (Gb/s)

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

## 6. Megatron-LM 1-node — reference ~7B GPT (TFLOP/s/GPU)

Same config on both: 36 layers, hidden 4096, FFN 14336, seq 2048, bf16, micro-batch 4, global batch = 128 x GPUs, 100 iters, no recompute, TP=PP=1. Metric = last-iteration throughput per GPU. B200 = mean over B200 nodes with that GPU count (newest run).

| #GPUs | GBS | B200 TFLOP/s/GPU | #B200 nodes | B300 TFLOP/s/GPU | B300 aggregate | B200 iter ms | B300 iter ms | B300 / B200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 989.0 | 1 | 1,021.2 | 1,021 | 11,374 | 11,016 | **1.03x** |
| 2 | 256 | 982.9 | 1 | 1,030.8 | 2,062 | 11,446 | 10,913 | **1.05x** |
| 3 | 384 | 969.8 | 1 | 1,025.3 | 3,076 | 11,599 | 10,972 | **1.06x** |
| 4 | 512 | 967.0 | 1 | 1,018.3 | 4,073 | 11,634 | 11,048 | **1.05x** |
| 5 | 640 | 966.7 | 1 | 1,015.0 | 5,075 | 11,637 | 11,084 | **1.05x** |
| 6 | 768 | 969.1 | 1 | 1,014.2 | 6,085 | 11,608 | 11,092 | **1.05x** |
| 7 | 896 | 968.6 | 1 | 1,018.3 | 7,128 | 11,615 | 11,047 | **1.05x** |
| 8 | 1024 | 968.8 | 3 | 1,003.2 | 8,026 | 11,612 | 11,214 | **1.04x** |

## 7. Megatron-LM 1-node — tuned max throughput, best per GPU type (TFLOP/s/GPU)

Each GPU type gets its own grid sweep (8 GPUs, seq 4096, distributed optimizer, overlapped grad-reduce/param-gather, TP=PP=1, grad-acc 4, 20 iters): 5B (24L, h4096) micro 4/8/16 x recompute none/selective/full, and 13B (40L, h5120) micro 2/4/8 x recompute none/full, each in bf16 and fp8. The configs are NOT forced to match: the best point per GPU type and precision is compared. TFLOP/s/GPU is Megatron's model-FLOP throughput (recompute work not counted). Peak memory = max allocated by PyTorch on rank 0.

| Precision | B200 best | B200 config | B200 mem GiB | B300 best | B300 config | B300 mem GiB | B300 / B200 |
|---|---:|---|---:|---:|---|---:|---:|
| bf16 | 1,017.6 | 13b mb2 none | 147 | 1,135.7 | 13b mb4 none | 202 | **1.12x** |
| fp8 | 1,499.7 | 13b mb2 none | 151 | 1,697.5 | 13b mb2 none | 151 | **1.13x** |

Grid points finished OK: B200 22/30, B300 28/30 (of 30 each).

### Full sweep grid (TFLOP/s/GPU, peak GiB)

| Prec | Model | Micro | Recompute | B200 | B300 | B300 / B200 |
|---|---|---:|---|---:|---:|---:|
| bf16 | 5b | 4 | none | 1,006.0 (92 GiB) | 1,080.8 (92 GiB) | **1.07x** |
| bf16 | 5b | 4 | selective | 980.2 (92 GiB) | 1,058.4 (92 GiB) | **1.08x** |
| bf16 | 5b | 4 | full | 774.5 (44 GiB) | 842.4 (44 GiB) | **1.09x** |
| bf16 | 5b | 8 | none | 1,012.9 (148 GiB) | 1,011.0 (148 GiB) | **1.00x** |
| bf16 | 5b | 8 | selective | 1,012.2 (148 GiB) | 1,046.1 (148 GiB) | **1.03x** |
| bf16 | 5b | 8 | full | 782.2 (52 GiB) | 796.6 (52 GiB) | **1.02x** |
| bf16 | 5b | 16 | none | OOM | 1,059.1 (260 GiB) | — |
| bf16 | 5b | 16 | selective | OOM | 1,054.1 (260 GiB) | — |
| bf16 | 5b | 16 | full | 797.7 (68 GiB) | 830.2 (68 GiB) | **1.04x** |
| bf16 | 13b | 2 | none | 1,017.6 (147 GiB) | 1,062.4 (147 GiB) | **1.04x** |
| bf16 | 13b | 2 | full | 793.9 (97 GiB) | 814.4 (97 GiB) | **1.03x** |
| bf16 | 13b | 4 | none | OOM | 1,135.7 (202 GiB) | — |
| bf16 | 13b | 4 | full | 824.1 (102 GiB) | 856.2 (102 GiB) | **1.04x** |
| bf16 | 13b | 8 | none | OOM | OOM | — |
| bf16 | 13b | 8 | full | 833.3 (114 GiB) | 880.9 (114 GiB) | **1.06x** |
| fp8 | 5b | 4 | none | 1,302.7 (89 GiB) | 1,325.1 (89 GiB) | **1.02x** |
| fp8 | 5b | 4 | selective | 1,307.4 (86 GiB) | 1,272.6 (86 GiB) | **0.97x** |
| fp8 | 5b | 4 | full | 1,083.4 (49 GiB) | 1,218.1 (49 GiB) | **1.12x** |
| fp8 | 5b | 8 | none | 1,404.0 (138 GiB) | 1,493.7 (138 GiB) | **1.06x** |
| fp8 | 5b | 8 | selective | 1,390.6 (132 GiB) | 1,427.5 (132 GiB) | **1.03x** |
| fp8 | 5b | 8 | full | 1,104.4 (57 GiB) | 1,201.0 (57 GiB) | **1.09x** |
| fp8 | 5b | 16 | none | OOM | 1,476.6 (235 GiB) | — |
| fp8 | 5b | 16 | selective | OOM | 1,524.1 (223 GiB) | — |
| fp8 | 5b | 16 | full | 1,120.6 (73 GiB) | 1,194.0 (73 GiB) | **1.07x** |
| fp8 | 13b | 2 | none | 1,499.7 (151 GiB) | 1,697.5 (151 GiB) | **1.13x** |
| fp8 | 13b | 2 | full | 1,129.5 (108 GiB) | 1,175.9 (108 GiB) | **1.04x** |
| fp8 | 13b | 4 | none | OOM | 1,585.6 (199 GiB) | — |
| fp8 | 13b | 4 | full | 1,205.7 (114 GiB) | 1,280.8 (114 GiB) | **1.06x** |
| fp8 | 13b | 8 | none | OOM | OOM | — |
| fp8 | 13b | 8 | full | 1,229.2 (125 GiB) | 1,276.2 (125 GiB) | **1.04x** |

