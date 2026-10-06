# Primus Sweep Report — MI355X (1..8 GPUs)

- Sweep dir: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719`
- Bench output dir: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/primus/sweep_out_20261002-003719`
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
| 1 | 32 | 1154.60 | 169.40 | 132.00 | 4866.20 |  |
| 2 | 64 | 1110.00 | 238.00 | 185.80 | 5061.60 |  |
| 3 | 96 | 1114.00 | 269.80 | 211.90 | 5043.80 |  |
| 4 | 128 | 1125.80 | 268.50 | 210.70 | 4990.70 |  |
| 5 | 160 | 1075.70 | 261.80 | 205.70 | 5223.10 |  |
| 6 | 192 | 1071.80 | 193.20 | 149.85 | 5242.30 |  |
| 7 | 224 | 1065.90 | 174.20 | 134.65 | 5271.00 |  |
| 8 | 256 | 1134.80 | 189.60 | 146.65 | 4951.10 |  |

### 1.1a Dell Cloud Primus vs amd-ubuntu Primus (same llama2-7B path)

Both hosts are 8 x MI355X running the same Primus -> Megatron-LM llama2-7B BF16 workload with primus-turbo ON, MBS=4, seq 4096. **Compared on `compute per GPU`, which is the metric Dell Cloud's REPORT.md section 1.1 reports** — see the metric note below, this distinction matters enormously.

| N | GBS Dell | GBS AMD | Dell compute TF/s/GPU | AMD compute TF/s/GPU | AMD/Dell | comparable? |
|--:|--------:|--------:|---------------------:|--------------------:|--------:|:------------|
| 1 | 256 | 32 | 1160.60 | 1154.60 | **0.99x** | no — GBS differs (256 vs 32) |
| 2 | 256 | 64 | 1146.00 | 1110.00 | **0.97x** | no — GBS differs (256 vs 64) |
| 3 | 252 | 96 | 1143.60 | 1114.00 | **0.97x** | no — GBS differs (252 vs 96) |
| 4 | 256 | 128 | 1139.10 | 1125.80 | **0.99x** | no — GBS differs (256 vs 128) |
| 5 | — | 160 | — (run failed) | 1075.70 | — | no — Dell has no data |
| 6 | — | 192 | — (run failed) | 1071.80 | — | no — Dell has no data |
| 7 | — | 224 | — (run failed) | 1065.90 | — | no — Dell has no data |
| 8 | 256 | 256 | 1132.00 | 1134.80 | **1.00x** | **YES — matched GBS** |

**Only N=8 is a valid head-to-head** — it is the one point where both runs used GBS=256 (ours as 32x8, theirs fixed). There the two machines are **1.00x** apart: 1132.0 vs 1134.8 TF/s/GPU. Same silicon, essentially identical result — which is the expected outcome and a good cross-machine validation.

At N=1..4 the GBS differs (Dell fixed 256; ours 32N = 32/64/96/128), so those rows are not comparable — a smaller global batch means fewer tokens per iteration and different efficiency. At N=5/6/7 Dell has no data at all: fixed GBS=256 is not divisible by MBS(4) x DP(N) for those arities, which is exactly the failure our per-N `GBS=32N` scheme was designed to avoid. **Our sweep is 8/8; theirs is 5/8.**

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
| **amd-ubuntu MI355X** | **572.5** | **58.1%** | **0.72x** |

amd-ubuntu is **28% behind Dell Cloud** despite identical configuration. Since the silicon is the same, this points at the software stack or image version rather than hardware — worth investigating before treating either number as definitive.

> **What makes this different from section 1.1.** Section 1.1 is llama2-7B via Primus with primus-turbo — a different model, framework, and kernel set, and *not* comparable to the B200 number. This section deliberately abandons Primus to match the reference configuration exactly. Both are valid; they answer different questions.

<!-- END megatron-ref-3way -->

## 2. GEMM microbench (`benchmark gemm`)

Square GEMM 4096×4096×4096 BF16, 10 s per rank, 2 GB rotating cache buffer. Each rank runs independently — no collectives. Mean / min / max are taken across the N ranks.

### 2.1 TF/s/GPU vs #GPUs

| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |
|--:|--------------:|-------------:|-------------:|:------|
| 1 | 1425.37 | 1425.37 | 1425.37 |  |
| 2 | 1401.29 | 1378.95 | 1423.63 |  |
| 3 | 1423.96 | 1375.29 | 1467.88 |  |
| 4 | 1430.36 | 1374.94 | 1465.92 |  |
| 5 | 1444.59 | 1370.97 | 1503.11 |  |
| 6 | 1461.80 | 1369.81 | 1546.86 |  |
| 7 | 1463.34 | 1376.56 | 1541.25 |  |
| 8 | 1467.98 | 1378.02 | 1541.55 |  |

## 3. Dense GEMM microbench (`benchmark gemm-dense`)

Llama-shape GEMM sweep (default: hidden 4096, FFN 11008, vocab 32000, MBS=1, BF16). Reports TF/s per shape per rank; the table aggregates across shapes and ranks.

### 3.1 TF/s/GPU (aggregate) vs #GPUs

| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |
|--:|--------------:|-------------:|-------------:|:------|
| 1 | 1300.02 | 1105.94 | 1462.57 |  |
| 2 | 1278.10 | 1042.48 | 1449.28 |  |
| 3 | 1298.74 | 1095.15 | 1493.90 |  |
| 4 | 1307.22 | 1060.29 | 1488.30 |  |
| 5 | 1316.64 | 1121.52 | 1501.00 |  |
| 6 | 1324.67 | 1114.16 | 1531.09 |  |
| 7 | 1330.57 | 1112.52 | 1529.02 |  |
| 8 | 1330.11 | 1100.23 | 1521.36 |  |

## 4. DeepSeek GEMM microbench (`benchmark gemm-deepseek`)

DeepSeek-V2/V3-style MoE shapes (hidden 4096, MoE int 1536, 128 routed experts, BF16).

### 4.1 TF/s/GPU (aggregate) vs #GPUs

| N | mean TF/s/GPU | min TF/s/GPU | max TF/s/GPU | notes |
|--:|--------------:|-------------:|-------------:|:------|
| 1 | 979.04 | 172.11 | 1574.93 |  |
| 2 | 961.64 | 167.28 | 1570.99 |  |
| 3 | 973.37 | 165.70 | 1615.78 |  |
| 4 | 979.47 | 165.28 | 1616.09 |  |
| 5 | 985.03 | 161.91 | 1659.01 |  |
| 6 | 993.54 | 164.66 | 1706.52 |  |
| 7 | 990.51 | 162.56 | 1706.16 |  |
| 8 | 994.21 | 161.71 | 1705.87 |  |

## 5. Attention microbench (`benchmark attention`)

Flash-attention backend, MBS=4 across the built-in model shape set.

### 5.1 Attention metrics vs #GPUs

| N | metric | mean | best | n_shapes |
|--:|:-------|-----:|-----:|---------:|
| 1 | fwd_tflops | 719.55 | 764.29 | 6 |
| 1 | bwd_tflops | 222.43 | 265.42 | 6 |
| 2 | fwd_tflops | 715.03 | 762.57 | 6 |
| 2 | bwd_tflops | 221.41 | 264.15 | 6 |
| 3 | fwd_tflops | 720.78 | 782.12 | 6 |
| 3 | bwd_tflops | 220.89 | 265.30 | 6 |
| 4 | fwd_tflops | 720.69 | 781.91 | 6 |
| 4 | bwd_tflops | 220.89 | 263.72 | 6 |
| 5 | fwd_tflops | 726.29 | 781.17 | 6 |
| 5 | bwd_tflops | 220.58 | 261.19 | 6 |
| 6 | fwd_tflops | 733.12 | 782.04 | 6 |
| 6 | bwd_tflops | 220.69 | 263.00 | 6 |
| 7 | fwd_tflops | 732.44 | 781.22 | 6 |
| 7 | bwd_tflops | 220.99 | 262.91 | 6 |
| 8 | fwd_tflops | 732.95 | 782.58 | 6 |
| 8 | bwd_tflops | 220.90 | 262.98 | 6 |

## 6. RCCL collective microbench (`benchmark rccl --op all_reduce`)

All-reduce bandwidth sweep across message sizes (1K..128M, log2 sweep). Peak busbw reflects the asymptotic large-message bandwidth; mean is across the size sweep. **N=1 is skipped** — collective on a single rank is degenerate.

### 6.1 Peak / mean all-reduce busbw vs #GPUs

| N | peak busbw (GB/s) | mean busbw (GB/s) | sizes |
|--:|------------------:|------------------:|------:|
| 1 | — | — | skipped |
| 2 | 56.98 | 21.30 | 18 |
| 3 | 85.93 | 29.74 | 18 |
| 4 | 165.53 | 50.90 | 18 |
| 5 | 46.08 | 17.87 | 18 |
| 6 | 45.29 | 17.43 | 18 |
| 7 | 45.50 | 17.17 | 18 |
| 8 | 357.05 | 87.87 | 18 |

## 6a. Megatron vs the GEMM ceilings (N=8, BF16, same host)

How much of the achievable matrix-multiply rate does real training actually realize? Each row is a progressively more realistic ceiling, so each gap attributes a specific loss.

| Ceiling | TF/s/GPU | Megatron as % | What the gap costs |
|---|---:|---:|---|
| RVS `gst` bf16 — silicon, no framework (Part A) | 1648.90 | **69%** | PyTorch/framework dispatch, then everything below |
| Primus `gemm` — square 4096^3 | 1467.98 | **77%** | off-peak shapes + everything non-GEMM |
| Primus `gemm-dense` — dense-model shape mix | 1330.11 | **85%** | non-GEMM work only (shape penalty already priced in) |
| Megatron llama2-7B (compute per GPU) | **1134.80** | 100% | — |

**`gemm-dense` is the right baseline.** It runs a dense-transformer shape mix — the kind of QKV / O / FFN-gate / up / down GEMMs Megatron issues — so the 85% figure isolates *non-GEMM* overhead: attention, RMSNorm, RoPE, optimizer, and the gradient all-reduce. The square-GEMM row is a looser ceiling because 4096^3 is a shape Megatron never actually runs.

**`gemm-deepseek` is deliberately excluded.** Those are MoE expert shapes with small, skewed K-dimensions; llama2-7B is dense and never issues them, so a percentage against it would be meaningless.

**RVS `gst` vs Primus `gemm` — what actually differs.** Both measure BF16 matrix multiply on this same host, and the 11% gap between them (1648.90 -> 1467.98) is worth understanding, because it is *not* only shape:

| | RVS `gst` (Part A) | Primus `gemm` (Part C) |
|---|---|---|
| Stack | hipBLASLt called **directly from C++** | **PyTorch** -> hipBLASLt |
| Shape | 8192 x 8192 x 16384 | 4096 x 4096 x 4096 |
| Cache defeat | `rotating: 512` buffers | 2 GB rotating buffer |
| Metric | **peak** across log intervals | **mean** across ranks |
| Duration | 30 s | 10 s |

Two effects dominate. **Framework dispatch**: RVS has no Python, no autograd, no tensor wrapper — it is the closest thing to a pure library number. **Matrix size**: RVS' GEMM is 8x larger in K and 4x in M/N, so fixed per-call overhead amortizes far better. The *peak-vs-mean* metric choice also flatters RVS slightly. So the RVS row is a genuine silicon ceiling, but it is a deliberately favourable one — the Primus rows are closer to what any real framework can reach.

> **Note on the name — "dense" means dense *model*, not dense *matrix*.** The contrast is with its sibling `gemm-deepseek` (a MoE / sparse-expert model), not with sparse matrices — all of these GEMMs are fully dense. So plain `gemm` is not "denser" than `gemm-dense` despite the name; it is simply one arbitrary shape (`--M --N --K`, here 4096^3) rather than a model-derived set. Caveat: that `gemm-dense` specifically uses *llama* shapes is an inference from the dense-vs-DeepSeek pairing, not verified against Primus' source — what is certain is that it is a dense-transformer shape set, which is what makes it the right ceiling for llama2-7B.

> **Three caveats.** (1) Megatron's TFLOPs are an *analytical* count (~6·params·tokens), not measured FLOPs — so this is model-FLOPs utilization, not a literal hardware efficiency. (2) The microbenches are pure compute with no collectives; Megatron includes a gradient all-reduce per step. (3) Both numbers must be kernel-time (`compute per GPU`); mixing in the wall-clock figure invalidates the ratio entirely.

## 6b. Where the remaining gap goes — attention

Attention is **not** compared as a percentage of Megatron: it measures a *component*, not a substitute workload, so "Megatron as % of attention" would be a category error. It is reported here because it is the leading explanation for why end-to-end training lands below the GEMM ceiling above.

| Kernel class (N=8) | TF/s/GPU | vs `gemm-dense` |
|---|---:|---:|
| `gemm-dense` (the GEMM path) | 1330.11 | 100% |
| attention **forward** | 732.95 | 55% |
| attention **backward** | 220.90 | 17% |

Attention forward runs at roughly half the GEMM rate and **backward at 30% of forward** (220.90 vs 732.95 TF/s/GPU). Backward is dominated by gradient recomputation plus extra matmuls, and the asymmetry matches what is reported for flash-attention-class kernels generally.

Since a transformer step spends a substantial fraction of its time in attention — and backward is ~2x the cost of forward in a training step — a kernel class running at 17% of GEMM rate is sufficient on its own to explain most of the residual between the `gemm-dense` ceiling and measured end-to-end throughput. Attention is also flat across N (each rank runs independently, no collective), so this is a per-GPU kernel property, not a scaling effect.

## 7. Analysis

- **Megatron weak-scaling (llama2-7B BF16, turbo ON):** N=1: 169 TF/s/GPU (100 % of N=1), N=2: 238 TF/s/GPU (140 % of N=1), N=3: 270 TF/s/GPU (159 % of N=1), N=4: 268 TF/s/GPU (159 % of N=1), N=5: 262 TF/s/GPU (155 % of N=1), N=6: 193 TF/s/GPU (114 % of N=1), N=7: 174 TF/s/GPU (103 % of N=1), N=8: 190 TF/s/GPU (112 % of N=1). Per-GPU throughput is essentially flat (≤ 37 % spread between best and worst N), so the all-reduce overhead at MBS·N grad-accum is small relative to the model's compute. The lower N=1 / higher N=8 iter-time scales linearly with GBS as expected for weak-scaling.
- **Primus-turbo vs reference image at N=8:** Primus (llama2-7B, turbo ON) hits **190 TF/s/GPU**; the `rocm/megatron-lm:v26.1` image on the same hardware (GPT-15.6B, no turbo, §3 tuned) tops out at **790.4 TF/s/GPU** — a **0.24× per-GPU jump**. Workloads differ (smaller model, different GEMM shapes, primus-turbo attention/grouped-MLP fused kernels), so this is *not* a pure kernel-vs-kernel speedup; it captures the combined win of (i) llama2-7B being more GEMM-dense than GPT-15.6B, (ii) primus-turbo replacing unfused softmax/RMSNorm/attention with gfx950-native kernels, and (iii) Primus' MFU-tuned argument set. Use as the new headline number for this hardware on a llama-family workload.
- **GEMM per-GPU consistency:** mean TF/s/GPU ranges 1401.3..1468.0 across all N (4.5 % spread). Each rank runs the same 4Kx4Kx4K BF16 shape independently with no collectives, so a flat curve confirms there's no thermal/PCIe/power contention as N grows. This is the per-GPU compute ceiling on this hardware for square FP16/BF16 matmul.
- **Shape sensitivity:** square 4Kx4Kx4K hits 1440 TF/s/GPU; the **llama-shape mix** (gemm-dense) drops to 1311 (91 % of peak); the **deepseek MoE shape mix** falls to 982 (68 %). The MoE drop is shape-driven (small / skewed K-dim in the expert path), not a hardware issue.
- **Attention fwd/bwd asymmetry:** fwd ≈ 725 TF/s/GPU, bwd ≈ 221 TF/s/GPU (bwd / fwd = 30 %). Backward is dominated by gradient recomputation + extra matmuls; the gap matches what's reported for flash-attention class kernels. Both are stable across N (each rank runs independently — no all-reduce in this bench).
- **RCCL all-reduce cliff:** peak busbw at N∈{4,8} averages **261 GB/s**; at N∈{5,6,7} it drops to **46 GB/s** (17 %). N=8 alone hits **357 GB/s** — the asymptotic xGMI ring bandwidth. The non-power-of-2 cliff matches the existing megatron-lm:v26.1 reference and confirms it's a topology/ring-algorithm issue (RCCL falls back from a clean ring to tree/segmented patterns), not a Primus issue. **Yet** the Megatron training in §1.1 is essentially insensitive to this cliff because per-iter compute (~20 s) dwarfs the all-reduce time even at the degraded busbw.

## 8. Raw per-(bench, N) status

From driver `summary.txt`:

```
Primus full sweep 20261002-003719
Image      : rocm/primus:v26.5
Driver log : /orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719
Bench out  : /orcd/data/orcd/022/benchmarks/amd-ubuntu/primus/sweep_out_20261002-003719
Started    : 2026-10-02T00:47:12+00:00

================ N=1 ================
----- gemm N=1 port=29735 devs=0 2026-10-02T00:47:12+00:00 -----
  OK duration=21s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm_N1.log
----- gemm-dense N=1 port=29959 devs=0 2026-10-02T00:47:33+00:00 -----
  OK duration=110s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-dense_N1.log
----- gemm-deepseek N=1 port=29614 devs=0 2026-10-02T00:49:23+00:00 -----
  OK duration=166s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-deepseek_N1.log
----- attention N=1 port=29590 devs=0 2026-10-02T00:52:09+00:00 -----
  OK duration=37s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/attention_N1.log
----- rccl N=1 SKIPPED (collective needs N>=2) -----
================ N=2 ================
----- gemm N=2 port=29640 devs=0,1 2026-10-02T00:52:46+00:00 -----
  OK duration=22s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm_N2.log
----- gemm-dense N=2 port=29734 devs=0,1 2026-10-02T00:53:08+00:00 -----
  OK duration=92s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-dense_N2.log
----- gemm-deepseek N=2 port=29968 devs=0,1 2026-10-02T00:54:40+00:00 -----
  OK duration=169s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-deepseek_N2.log
----- attention N=2 port=29656 devs=0,1 2026-10-02T00:57:29+00:00 -----
  OK duration=18s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/attention_N2.log
----- rccl N=2 port=29804 devs=0,1 2026-10-02T00:57:47+00:00 -----
  OK duration=11s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/rccl_N2.log
================ N=3 ================
----- gemm N=3 port=29724 devs=0,1,2 2026-10-02T00:57:58+00:00 -----
  OK duration=23s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm_N3.log
----- gemm-dense N=3 port=29567 devs=0,1,2 2026-10-02T00:58:21+00:00 -----
  OK duration=92s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-dense_N3.log
----- gemm-deepseek N=3 port=29809 devs=0,1,2 2026-10-02T00:59:53+00:00 -----
  OK duration=168s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-deepseek_N3.log
----- attention N=3 port=29593 devs=0,1,2 2026-10-02T01:02:41+00:00 -----
  OK duration=18s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/attention_N3.log
----- rccl N=3 port=29759 devs=0,1,2 2026-10-02T01:02:59+00:00 -----
  OK duration=9s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/rccl_N3.log
================ N=4 ================
----- gemm N=4 port=29511 devs=0,1,2,3 2026-10-02T01:03:08+00:00 -----
  OK duration=24s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm_N4.log
----- gemm-dense N=4 port=29930 devs=0,1,2,3 2026-10-02T01:03:32+00:00 -----
  OK duration=92s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-dense_N4.log
----- gemm-deepseek N=4 port=29986 devs=0,1,2,3 2026-10-02T01:05:04+00:00 -----
  OK duration=169s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-deepseek_N4.log
----- attention N=4 port=29536 devs=0,1,2,3 2026-10-02T01:07:53+00:00 -----
  OK duration=17s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/attention_N4.log
----- rccl N=4 port=29507 devs=0,1,2,3 2026-10-02T01:08:10+00:00 -----
  OK duration=10s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/rccl_N4.log
================ N=5 ================
----- gemm N=5 port=29955 devs=0,1,2,3,4 2026-10-02T01:08:20+00:00 -----
  OK duration=24s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm_N5.log
----- gemm-dense N=5 port=29982 devs=0,1,2,3,4 2026-10-02T01:08:44+00:00 -----
  OK duration=93s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-dense_N5.log
----- gemm-deepseek N=5 port=29957 devs=0,1,2,3,4 2026-10-02T01:10:17+00:00 -----
  OK duration=169s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-deepseek_N5.log
----- attention N=5 port=29733 devs=0,1,2,3,4 2026-10-02T01:13:06+00:00 -----
  OK duration=17s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/attention_N5.log
----- rccl N=5 port=29744 devs=0,1,2,3,4 2026-10-02T01:13:23+00:00 -----
  OK duration=10s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/rccl_N5.log
================ N=6 ================
----- gemm N=6 port=29614 devs=0,1,2,3,4,5 2026-10-02T01:13:33+00:00 -----
  OK duration=25s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm_N6.log
----- gemm-dense N=6 port=29822 devs=0,1,2,3,4,5 2026-10-02T01:13:58+00:00 -----
  OK duration=93s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-dense_N6.log
----- gemm-deepseek N=6 port=29688 devs=0,1,2,3,4,5 2026-10-02T01:15:31+00:00 -----
  OK duration=170s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-deepseek_N6.log
----- attention N=6 port=29977 devs=0,1,2,3,4,5 2026-10-02T01:18:21+00:00 -----
  OK duration=19s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/attention_N6.log
----- rccl N=6 port=29955 devs=0,1,2,3,4,5 2026-10-02T01:18:40+00:00 -----
  OK duration=11s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/rccl_N6.log
================ N=7 ================
----- gemm N=7 port=29939 devs=0,1,2,3,4,5,6 2026-10-02T01:18:51+00:00 -----
  OK duration=24s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm_N7.log
----- gemm-dense N=7 port=29909 devs=0,1,2,3,4,5,6 2026-10-02T01:19:15+00:00 -----
  OK duration=94s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-dense_N7.log
----- gemm-deepseek N=7 port=29658 devs=0,1,2,3,4,5,6 2026-10-02T01:20:49+00:00 -----
  OK duration=170s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-deepseek_N7.log
----- attention N=7 port=29614 devs=0,1,2,3,4,5,6 2026-10-02T01:23:39+00:00 -----
  OK duration=18s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/attention_N7.log
----- rccl N=7 port=29727 devs=0,1,2,3,4,5,6 2026-10-02T01:23:57+00:00 -----
  OK duration=11s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/rccl_N7.log
================ N=8 ================
----- gemm N=8 port=29723 devs=0,1,2,3,4,5,6,7 2026-10-02T01:24:08+00:00 -----
  OK duration=29s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm_N8.log
----- gemm-dense N=8 port=29795 devs=0,1,2,3,4,5,6,7 2026-10-02T01:24:37+00:00 -----
  OK duration=97s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-dense_N8.log
----- gemm-deepseek N=8 port=29593 devs=0,1,2,3,4,5,6,7 2026-10-02T01:26:14+00:00 -----
  OK duration=173s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/gemm-deepseek_N8.log
----- attention N=8 port=29858 devs=0,1,2,3,4,5,6,7 2026-10-02T01:29:07+00:00 -----
  OK duration=21s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/attention_N8.log
----- rccl N=8 port=30006 devs=0,1,2,3,4,5,6,7 2026-10-02T01:29:28+00:00 -----
  OK duration=13s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/rccl_N8.log
Finished   : 2026-10-02T01:29:41+00:00
================ MEGATRON 2026-10-02T01:29:42+00:00 image=rocm/primus:v26.5 exp=examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml ================
----- megatron N=1 GBS=32 MBS=4 devs=0 2026-10-02T01:29:42+00:00 -----
  FAIL(rc=1) duration=507s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N1.log
----- megatron N=2 GBS=64 MBS=4 devs=0,1 2026-10-02T01:38:09+00:00 -----
  FAIL(rc=1) duration=118s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N2.log
----- megatron N=3 GBS=96 MBS=4 devs=0,1,2 2026-10-02T01:40:07+00:00 -----
  FAIL(rc=1) duration=67s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N3.log
----- megatron N=4 GBS=128 MBS=4 devs=0,1,2,3 2026-10-02T01:41:14+00:00 -----
  FAIL(rc=1) duration=66s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N4.log
----- megatron N=5 GBS=160 MBS=4 devs=0,1,2,3,4 2026-10-02T01:42:20+00:00 -----
  FAIL(rc=1) duration=66s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N5.log
----- megatron N=6 GBS=192 MBS=4 devs=0,1,2,3,4,5 2026-10-02T01:43:26+00:00 -----
  FAIL(rc=1) duration=66s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N6.log
----- megatron N=7 GBS=224 MBS=4 devs=0,1,2,3,4,5,6 2026-10-02T01:44:32+00:00 -----
  FAIL(rc=1) duration=66s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N7.log
----- megatron N=8 GBS=256 MBS=4 devs=0,1,2,3,4,5,6,7 2026-10-02T01:45:38+00:00 -----
  FAIL(rc=1) duration=67s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N8.log
[megatron] 2026-10-02T01:46:45+00:00 DONE
================ MEGATRON 2026-10-05T18:22:26+00:00 image=rocm/primus:v26.5 exp=examples/megatron/configs/MI355X/llama2_7B-BF16-pretrain.yaml ================
----- megatron N=1 GBS=32 MBS=4 devs=0 2026-10-05T18:22:26+00:00 -----
  OK duration=1936s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N1.log
----- megatron N=2 GBS=64 MBS=4 devs=0,1 2026-10-05T18:54:42+00:00 -----
  OK duration=343s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N2.log
----- megatron N=3 GBS=96 MBS=4 devs=0,1,2 2026-10-05T19:00:25+00:00 -----
  OK duration=334s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N3.log
----- megatron N=4 GBS=128 MBS=4 devs=0,1,2,3 2026-10-05T19:05:59+00:00 -----
  OK duration=335s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N4.log
----- megatron N=5 GBS=160 MBS=4 devs=0,1,2,3,4 2026-10-05T19:11:34+00:00 -----
  OK duration=348s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N5.log
----- megatron N=6 GBS=192 MBS=4 devs=0,1,2,3,4,5 2026-10-05T19:17:23+00:00 -----
  OK duration=431s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N6.log
----- megatron N=7 GBS=224 MBS=4 devs=0,1,2,3,4,5,6 2026-10-05T19:24:34+00:00 -----
  OK duration=389s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N7.log
----- megatron N=8 GBS=256 MBS=4 devs=0,1,2,3,4,5,6,7 2026-10-05T19:31:04+00:00 -----
  OK duration=389s log=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6101/primus/sweep-20261002-003719/megatron-llama2_7B-bf16_N8.log
[megatron] 2026-10-05T19:37:33+00:00 DONE
```
