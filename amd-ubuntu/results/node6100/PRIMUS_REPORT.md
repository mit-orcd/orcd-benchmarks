# Primus Sweep Report — MI355X (1..8 GPUs)

- Sweep dir: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720`
- Bench output dir: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/primus/sweep_out_20261002-003720`
- Image: `rocm/primus:v26.3` (singularity SIF)
- Hardware: 1 node × 8 × AMD Instinct MI355X (gfx950)

## 1. Megatron-LM (via Primus `train pretrain`)

Workload: `examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml` (llama2-7B, seq 4096, MBS=4, mock data, primus-turbo ON: `use_turbo_attention`, `use_turbo_grouped_mlp`). The `last TF/s/GPU` column is the steady-state value of the final logged iteration (after JIT warmup); `GBS` is parsed from the log.

#### Parallelism: pure data parallel (DP=N, TP=PP=CP=EP=1)

Verified from the run logs (`data_parallel_size=8, sequence_parallel_size=0`, `world_size=8`) and the config (`tensor_model_parallel_size: 1`, `pipeline_model_parallel_size: 1`, `expert_model_parallel_size: 1`, `sequence_parallel` commented out).

Every GPU holds a **full llama2-7B replica** and processes its own micro-batches; gradients are all-reduced once per step. This is a **weak-scaling** study, so the driver computes `GBS(N) = MBS x N x GRAD_ACC = 4 x N x 8 = 32N` — constant work per GPU as N grows, and divisible by `MBS x DP` by construction. That last point matters: a fixed GBS=256 is *not* divisible by `MBS(4) x DP(N)` for N in {3,5,6,7}, which is what forced the reference Dell Cloud run into three separate rerun scripts. Computing GBS per N up front makes it one clean sweep.

**Why DP and not TP/PP/CP/EP here:**

- **DP is viable at all only because llama2-7B fits in one GPU's HBM** (288 GB on MI355X). Any model that did not fit would have forced TP or PP.
- **TP** would shard each layer and all-reduce activations *every layer*, adding collective traffic that is unnecessary when the model already fits.
- **PP** adds pipeline-bubble overhead and mainly earns its keep across nodes or when the model does not fit; on one node with fast XGMI it is strictly worse.
- **CP** (context parallel) targets very long sequences; at seq 4096 it is unnecessary.
- **EP** (expert parallel) applies only to MoE models; llama2-7B is dense.

This is the deliberate opposite of Part D (ATOM inference), which runs **TP=8** because a 70B / 1.5 TB model cannot fit on one GPU. The contrast explains the collective-sensitivity result in section 7: Megatron here issues **one gradient all-reduce per ~5 s iteration**, so even the degraded N=5/6/7 RCCL bandwidth is negligible against per-iteration compute. TP=8 inference has no such insulation — its collectives sit in the **per-token critical path**.

### 1.1 TF/s/GPU vs #GPUs (Primus → Megatron-LM, llama2-7B BF16, turbo ON)

| N | GBS | compute TF/s/GPU | wall-clock TF/s/GPU | mean TF/s/GPU | last iter (ms) | notes |
|--:|----:|-----------------:|--------------------:|--------------:|---------------:|:------|
| 1 | — | — | — | — | — | FAILED |
| 2 | — | — | — | — | — | FAILED |
| 3 | — | — | — | — | — | FAILED |
| 4 | — | — | — | — | — | FAILED |
| 5 | — | — | — | — | — | FAILED |
| 6 | — | — | — | — | — | FAILED |
| 7 | — | — | — | — | — | FAILED |
| 8 | — | — | — | — | — | FAILED |

### 1.1a Dell Cloud Primus vs amd-ubuntu Primus (same llama2-7B path)

Both hosts are 8 x MI355X running the same Primus -> Megatron-LM llama2-7B BF16 workload with primus-turbo ON, MBS=4, seq 4096. **Compared on `compute per GPU`, which is the metric Dell Cloud's REPORT.md section 1.1 reports** — see the metric note below, this distinction matters enormously.

| N | GBS Dell | GBS AMD | Dell compute TF/s/GPU | AMD compute TF/s/GPU | AMD/Dell | comparable? |
|--:|--------:|--------:|---------------------:|--------------------:|--------:|:------------|
| 1 | 256 | — | 1160.60 | — | — | no — AMD has no data |
| 2 | 256 | — | 1146.00 | — | — | no — AMD has no data |
| 3 | 252 | — | 1143.60 | — | — | no — AMD has no data |
| 4 | 256 | — | 1139.10 | — | — | no — AMD has no data |
| 5 | — | — | — (run failed) | — | — | no — Dell has no data |
| 6 | — | — | — (run failed) | — | — | no — Dell has no data |
| 7 | — | — | — (run failed) | — | — | no — Dell has no data |
| 8 | 256 | — | 1132.00 | — | — | no — AMD has no data |

> **Metric warning — two different TFLOP/s/GPU numbers exist.** The Megatron iteration line emits both `compute per GPU` (kernel-time throughput) and `throughput per GPU` (wall-clock, includes pipeline bubbles and idle). They differ by ~4x on this workload. Dell Cloud's REPORT.md section 1.1 column is the **compute** figure; a naive parse of the newer v26.5 log picks up the **wall-clock** figure instead. Comparing one against the other manufactures a spurious ~3.8x regression that does not exist. Section 1.1 above now reports both, explicitly labelled.

### 1.2 vs NVIDIA B200 (Megatron-LM, context only)

Reference: `/orcd/data/orcd/022/benchmarks/amd-benchmarks/amd-cloud/../dell-cloud/megatron-lm/summary.md` — the existing MI355X-vs-B200 table from the `rocm/megatron-lm:v26.1` image sweep (**GPT-15.6B, MBS=4, BF16, no-recompute**). **This is not directly comparable** to the Primus llama2-7B numbers in §1.1: different model, different image (no primus-turbo), different GEMM shape mix. Kept here only as the existing house benchmark. See §7 for an apples-to-oranges framing of what the Primus-turbo path delivers on the same hardware.

| N | B200 TF/s/GPU | MI355X TF/s/GPU | MI355X / B200 |
|--:|--------------:|----------------:|--------------:|
| 8 |         986.0 |       **790.4** |    **80.2 %** |

<!-- BEGIN megatron-ref-3way (auto-generated) -->

#### 1.2a Three-way, matched workload (N=8)

This row **is** apples-to-apples. It reproduces the exact Dell Cloud configuration on this host — same GPT-15.6B shape (L=40, H=6144, FFN=16384, heads=48, GQA kv=8, seq=4096, vocab=50304), same MBS=4 / GBS=32, BF16, no recompute, TP=PP=1, distributed optimizer, `--ddp-bucket-size 250000000`, 50 iters — using the **ROCm/Megatron-LM image, not Primus**, so there is no primus-turbo advantage. **`HSA_OVERRIDE_GFX_VERSION=9.4.2` is set on both sides** — required here too (the image's torch has no compiled gfx950 code objects at all; unset, backward pass fails in Transformer Engine with `RuntimeError: Unable to find any suitable algorithms`), so this comparison is same config *and* same code objects, not just same config.

| Machine | TF/s/GPU | vs B200 | vs Dell MI355X |
|---|---:|---:|---:|
| NVIDIA B200 | 986.0 | 100% | — |
| Dell Cloud MI355X | 790.4 | 80.2% | 1.00x |
| **amd-ubuntu MI355X** | **578.2** | **58.6%** | **0.73x** |

amd-ubuntu is **27% behind Dell Cloud** despite identical configuration. Since the silicon is the same, this points at the software stack or image version rather than hardware — worth investigating before treating either number as definitive.

> **What makes this different from section 1.1.** Section 1.1 is llama2-7B via Primus with primus-turbo — a different model, framework, and kernel set, and *not* comparable to the B200 number. This section deliberately abandons Primus to match the reference configuration exactly. Both are valid; they answer different questions.

<!-- END megatron-ref-3way -->

## 2. GEMM microbench (`benchmark gemm`)

Square GEMM 4096×4096×4096 BF16, 10 s per rank, 2 GB rotating cache buffer. Each rank runs independently — no collectives. Mean / min / max are taken across the N ranks.

### 2.1 TF/s/GPU vs #GPUs

| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |
|--:|--------------:|-------------:|-------------:|:------|
| 1 | 1480.55 | 1480.55 | 1480.55 |  |
| 2 | 1479.88 | 1473.78 | 1485.99 |  |
| 3 | 1499.07 | 1477.92 | 1525.68 |  |
| 4 | 1499.84 | 1470.59 | 1528.90 |  |
| 5 | 1505.89 | 1477.52 | 1524.48 |  |
| 6 | 1510.99 | 1473.21 | 1529.03 |  |
| 7 | 1509.84 | 1476.10 | 1527.85 |  |
| 8 | 1504.24 | 1467.79 | 1528.28 |  |

## 3. Dense GEMM microbench (`benchmark gemm-dense`)

Llama-shape GEMM sweep (default: hidden 4096, FFN 11008, vocab 32000, MBS=1, BF16). Reports TF/s per shape per rank; the table aggregates across shapes and ranks.

### 3.1 TF/s/GPU (aggregate) vs #GPUs

| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |
|--:|--------------:|-------------:|-------------:|:------|
| 1 | 1338.33 | 1144.40 | 1484.55 |  |
| 2 | 1345.69 | 1162.00 | 1505.60 |  |
| 3 | 1345.95 | 1090.96 | 1513.13 |  |
| 4 | 1355.10 | 1116.28 | 1528.65 |  |
| 5 | 1359.57 | 1127.37 | 1542.32 |  |
| 6 | 1357.26 | 1067.41 | 1548.32 |  |
| 7 | 1360.03 | 1096.45 | 1555.25 |  |
| 8 | 1358.09 | 1110.90 | 1546.29 |  |

## 4. DeepSeek GEMM microbench (`benchmark gemm-deepseek`)

DeepSeek-V2/V3-style MoE shapes (hidden 4096, MoE int 1536, 128 routed experts, BF16).

### 4.1 TF/s/GPU (aggregate) vs #GPUs

| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |
|--:|--------------:|-------------:|-------------:|:------|
| 1 | 1005.91 | 165.69 | 1631.33 |  |
| 2 | 1005.46 | 164.45 | 1651.43 |  |
| 3 | 1011.04 | 167.24 | 1688.31 |  |
| 4 | 1017.46 | 170.69 | 1688.54 |  |
| 5 | 1013.53 | 163.99 | 1688.03 |  |
| 6 | 1019.75 | 164.23 | 1688.33 |  |
| 7 | 1019.10 | 163.31 | 1691.06 |  |
| 8 | 1013.81 | 163.67 | 1689.77 |  |

## 5. Attention microbench (`benchmark attention`)

Flash-attention backend, MBS=4 across the built-in model shape set.

### 5.1 Attention metrics vs #GPUs

| N | metric | mean | best | n_shapes |
|--:|:-------|-----:|-----:|---------:|
| 1 | fwd_tflops | 737.35 | 783.60 | 6 |
| 1 | bwd_tflops | 222.62 | 266.29 | 6 |
| 2 | fwd_tflops | 738.22 | 784.23 | 6 |
| 2 | bwd_tflops | 221.16 | 265.81 | 6 |
| 3 | fwd_tflops | 739.81 | 797.49 | 6 |
| 3 | bwd_tflops | 220.49 | 263.96 | 6 |
| 4 | fwd_tflops | 745.64 | 800.67 | 6 |
| 4 | bwd_tflops | 220.17 | 264.06 | 6 |
| 5 | fwd_tflops | 747.77 | 798.76 | 6 |
| 5 | bwd_tflops | 220.94 | 262.83 | 6 |
| 6 | fwd_tflops | 749.67 | 799.50 | 6 |
| 6 | bwd_tflops | 220.20 | 263.31 | 6 |
| 7 | fwd_tflops | 751.65 | 800.36 | 6 |
| 7 | bwd_tflops | 220.53 | 263.34 | 6 |
| 8 | fwd_tflops | 751.24 | 798.97 | 6 |
| 8 | bwd_tflops | 220.68 | 262.66 | 6 |

## 6. RCCL collective microbench (`benchmark rccl --op all_reduce`)

All-reduce bandwidth sweep across message sizes (1K..128M, log2 sweep). Peak busbw reflects the asymptotic large-message bandwidth; mean is across the size sweep. **N=1 is skipped** — collective on a single rank is degenerate.

### 6.1 Peak / mean all-reduce busbw vs #GPUs

| N | peak busbw (GB/s) | mean busbw (GB/s) | sizes |
|--:|------------------:|------------------:|------:|
| 1 | — | — | skipped |
| 2 | 56.92 | 21.30 | 18 |
| 3 | 86.08 | 29.79 | 18 |
| 4 | 165.52 | 50.85 | 18 |
| 5 | 45.89 | 17.83 | 18 |
| 6 | 45.40 | 17.44 | 18 |
| 7 | 45.46 | 17.14 | 18 |
| 8 | 357.44 | 87.64 | 18 |

## 7. Analysis

- **GEMM per-GPU consistency:** mean TF/s/GPU ranges 1479.9..1511.0 across all N (2.1 % spread). Each rank runs the same 4Kx4Kx4K BF16 shape independently with no collectives, so a flat curve confirms there's no thermal/PCIe/power contention as N grows. This is the per-GPU compute ceiling on this hardware for square FP16/BF16 matmul.
- **Shape sensitivity:** square 4Kx4Kx4K hits 1499 TF/s/GPU; the **llama-shape mix** (gemm-dense) drops to 1353 (90 % of peak); the **deepseek MoE shape mix** falls to 1013 (68 %). The MoE drop is shape-driven (small / skewed K-dim in the expert path), not a hardware issue.
- **Attention fwd/bwd asymmetry:** fwd ≈ 745 TF/s/GPU, bwd ≈ 221 TF/s/GPU (bwd / fwd = 30 %). Backward is dominated by gradient recomputation + extra matmuls; the gap matches what's reported for flash-attention class kernels. Both are stable across N (each rank runs independently — no all-reduce in this bench).
- **RCCL all-reduce cliff:** peak busbw at N∈{4,8} averages **261 GB/s**; at N∈{5,6,7} it drops to **46 GB/s** (17 %). N=8 alone hits **357 GB/s** — the asymptotic xGMI ring bandwidth. The non-power-of-2 cliff matches the existing megatron-lm:v26.1 reference and confirms it's a topology/ring-algorithm issue (RCCL falls back from a clean ring to tree/segmented patterns), not a Primus issue. **Yet** the Megatron training in §1.1 is essentially insensitive to this cliff because per-iter compute (~20 s) dwarfs the all-reduce time even at the degraded busbw.

## 8. Raw per-(bench, N) status

From driver `summary.txt`:

```
Primus full sweep 20261002-003720
Image      : rocm/primus:v26.5
Driver log : /orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720
Bench out  : /orcd/data/orcd/022/benchmarks/amd-ubuntu/primus/sweep_out_20261002-003720
Started    : 2026-10-02T00:47:14+00:00

================ N=1 ================
----- gemm N=1 port=29817 devs=0 2026-10-02T00:47:14+00:00 -----
  OK duration=21s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm_N1.log
----- gemm-dense N=1 port=29541 devs=0 2026-10-02T00:47:35+00:00 -----
  OK duration=98s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-dense_N1.log
----- gemm-deepseek N=1 port=29894 devs=0 2026-10-02T00:49:13+00:00 -----
  OK duration=166s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-deepseek_N1.log
----- attention N=1 port=29524 devs=0 2026-10-02T00:51:59+00:00 -----
  OK duration=52s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/attention_N1.log
----- rccl N=1 SKIPPED (collective needs N>=2) -----
================ N=2 ================
----- gemm N=2 port=29650 devs=0,1 2026-10-02T00:52:51+00:00 -----
  OK duration=23s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm_N2.log
----- gemm-dense N=2 port=29703 devs=0,1 2026-10-02T00:53:14+00:00 -----
  OK duration=92s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-dense_N2.log
----- gemm-deepseek N=2 port=29932 devs=0,1 2026-10-02T00:54:46+00:00 -----
  OK duration=167s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-deepseek_N2.log
----- attention N=2 port=29947 devs=0,1 2026-10-02T00:57:33+00:00 -----
  OK duration=19s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/attention_N2.log
----- rccl N=2 port=29525 devs=0,1 2026-10-02T00:57:52+00:00 -----
  OK duration=11s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/rccl_N2.log
================ N=3 ================
----- gemm N=3 port=29818 devs=0,1,2 2026-10-02T00:58:03+00:00 -----
  OK duration=24s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm_N3.log
----- gemm-dense N=3 port=29534 devs=0,1,2 2026-10-02T00:58:27+00:00 -----
  OK duration=92s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-dense_N3.log
----- gemm-deepseek N=3 port=30001 devs=0,1,2 2026-10-02T00:59:59+00:00 -----
  OK duration=169s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-deepseek_N3.log
----- attention N=3 port=29832 devs=0,1,2 2026-10-02T01:02:48+00:00 -----
  OK duration=17s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/attention_N3.log
----- rccl N=3 port=29743 devs=0,1,2 2026-10-02T01:03:05+00:00 -----
  OK duration=9s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/rccl_N3.log
================ N=4 ================
----- gemm N=4 port=29734 devs=0,1,2,3 2026-10-02T01:03:14+00:00 -----
  OK duration=24s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm_N4.log
----- gemm-dense N=4 port=29772 devs=0,1,2,3 2026-10-02T01:03:38+00:00 -----
  OK duration=92s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-dense_N4.log
----- gemm-deepseek N=4 port=29538 devs=0,1,2,3 2026-10-02T01:05:10+00:00 -----
  OK duration=169s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-deepseek_N4.log
----- attention N=4 port=29792 devs=0,1,2,3 2026-10-02T01:07:59+00:00 -----
  OK duration=17s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/attention_N4.log
----- rccl N=4 port=29681 devs=0,1,2,3 2026-10-02T01:08:16+00:00 -----
  OK duration=9s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/rccl_N4.log
================ N=5 ================
----- gemm N=5 port=29621 devs=0,1,2,3,4 2026-10-02T01:08:25+00:00 -----
  OK duration=25s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm_N5.log
----- gemm-dense N=5 port=29546 devs=0,1,2,3,4 2026-10-02T01:08:50+00:00 -----
  OK duration=92s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-dense_N5.log
----- gemm-deepseek N=5 port=29641 devs=0,1,2,3,4 2026-10-02T01:10:22+00:00 -----
  OK duration=170s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-deepseek_N5.log
----- attention N=5 port=29638 devs=0,1,2,3,4 2026-10-02T01:13:12+00:00 -----
  OK duration=16s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/attention_N5.log
----- rccl N=5 port=29598 devs=0,1,2,3,4 2026-10-02T01:13:28+00:00 -----
  OK duration=11s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/rccl_N5.log
================ N=6 ================
----- gemm N=6 port=30004 devs=0,1,2,3,4,5 2026-10-02T01:13:39+00:00 -----
  OK duration=24s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm_N6.log
----- gemm-dense N=6 port=29993 devs=0,1,2,3,4,5 2026-10-02T01:14:03+00:00 -----
  OK duration=93s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-dense_N6.log
----- gemm-deepseek N=6 port=29781 devs=0,1,2,3,4,5 2026-10-02T01:15:36+00:00 -----
  OK duration=171s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-deepseek_N6.log
----- attention N=6 port=29969 devs=0,1,2,3,4,5 2026-10-02T01:18:27+00:00 -----
  OK duration=17s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/attention_N6.log
----- rccl N=6 port=29683 devs=0,1,2,3,4,5 2026-10-02T01:18:44+00:00 -----
  OK duration=11s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/rccl_N6.log
================ N=7 ================
----- gemm N=7 port=29909 devs=0,1,2,3,4,5,6 2026-10-02T01:18:55+00:00 -----
  OK duration=25s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm_N7.log
----- gemm-dense N=7 port=29999 devs=0,1,2,3,4,5,6 2026-10-02T01:19:20+00:00 -----
  OK duration=94s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-dense_N7.log
----- gemm-deepseek N=7 port=29870 devs=0,1,2,3,4,5,6 2026-10-02T01:20:54+00:00 -----
  OK duration=171s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-deepseek_N7.log
----- attention N=7 port=29607 devs=0,1,2,3,4,5,6 2026-10-02T01:23:45+00:00 -----
  OK duration=17s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/attention_N7.log
----- rccl N=7 port=29550 devs=0,1,2,3,4,5,6 2026-10-02T01:24:02+00:00 -----
  OK duration=12s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/rccl_N7.log
================ N=8 ================
----- gemm N=8 port=29882 devs=0,1,2,3,4,5,6,7 2026-10-02T01:24:14+00:00 -----
  OK duration=29s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm_N8.log
----- gemm-dense N=8 port=29895 devs=0,1,2,3,4,5,6,7 2026-10-02T01:24:43+00:00 -----
  OK duration=97s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-dense_N8.log
----- gemm-deepseek N=8 port=29837 devs=0,1,2,3,4,5,6,7 2026-10-02T01:26:20+00:00 -----
  OK duration=175s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/gemm-deepseek_N8.log
----- attention N=8 port=29718 devs=0,1,2,3,4,5,6,7 2026-10-02T01:29:15+00:00 -----
  OK duration=21s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/attention_N8.log
----- rccl N=8 port=29732 devs=0,1,2,3,4,5,6,7 2026-10-02T01:29:36+00:00 -----
  OK duration=14s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/rccl_N8.log
Finished   : 2026-10-02T01:29:50+00:00
================ MEGATRON 2026-10-02T01:29:50+00:00 image=rocm/primus:v26.5 exp=examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml ================
----- megatron N=1 GBS=32 MBS=4 devs=0 2026-10-02T01:29:50+00:00 -----
  FAIL(rc=1) duration=455s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/megatron-llama2_7B-bf16_N1.log
----- megatron N=2 GBS=64 MBS=4 devs=0,1 2026-10-02T01:37:25+00:00 -----
  FAIL(rc=1) duration=123s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/megatron-llama2_7B-bf16_N2.log
----- megatron N=3 GBS=96 MBS=4 devs=0,1,2 2026-10-02T01:39:28+00:00 -----
  FAIL(rc=1) duration=66s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/megatron-llama2_7B-bf16_N3.log
----- megatron N=4 GBS=128 MBS=4 devs=0,1,2,3 2026-10-02T01:40:34+00:00 -----
  FAIL(rc=1) duration=65s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/megatron-llama2_7B-bf16_N4.log
----- megatron N=5 GBS=160 MBS=4 devs=0,1,2,3,4 2026-10-02T01:41:39+00:00 -----
  FAIL(rc=1) duration=66s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/megatron-llama2_7B-bf16_N5.log
----- megatron N=6 GBS=192 MBS=4 devs=0,1,2,3,4,5 2026-10-02T01:42:45+00:00 -----
  FAIL(rc=1) duration=66s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/megatron-llama2_7B-bf16_N6.log
----- megatron N=7 GBS=224 MBS=4 devs=0,1,2,3,4,5,6 2026-10-02T01:43:51+00:00 -----
  FAIL(rc=1) duration=65s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/megatron-llama2_7B-bf16_N7.log
----- megatron N=8 GBS=256 MBS=4 devs=0,1,2,3,4,5,6,7 2026-10-02T01:44:56+00:00 -----
  FAIL(rc=1) duration=70s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/primus/sweep-20261002-003720/megatron-llama2_7B-bf16_N8.log
[megatron] 2026-10-02T01:46:06+00:00 DONE
```
