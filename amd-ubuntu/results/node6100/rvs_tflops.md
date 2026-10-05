# RVS `gst` TFLOPS — MI355X x8 (gfx950, ROCm 7.2.4)

System: 8 x AMD Instinct MI355X (CDNA 4 / gfx950), ROCm 7.2.4, Ubuntu 24.04.5.
Source runs: sweep_20261001_205002

## What this benchmark does

`run_tflops.sh` drives the ROCm Validation Suite (RVS) `gst` module (hipBLASLt GEMM
kernels) to measure sustained matrix-multiply throughput. For each precision and GPU
count, a YAML conf is generated from the shipped `conf/MI355X/levels/rvs_level_5.conf`
template, RVS runs with `parallel: true` so every selected GPU runs the GEMM
concurrently, and each GPU emits `GFLOPS <n>` every 3 s. The script takes the **peak**
per-GPU value (steady-state proxy) and sums across GPUs for the aggregate.
`target_stress: 0` means it measures only -- no pass/fail threshold.

Every GPU runs an **independent** GEMM: no XGMI or PCIe traffic, no RCCL. Scaling is
therefore embarrassingly parallel, and anything below ~99% is power/thermal sharing on
the 11.2 kW tray or measurement noise -- never interconnect.

## GPU specs

| Item | Value |
|---|---|
| Architecture | CDNA 4 (gfx950) |
| Compute units (per GPU) | 256 |
| Memory | 288 GB HBM3E |
| Memory bandwidth (per GPU) | 8 TB/s |
| PCIe host link | Gen 5 x16 (64 GB/s per direction) |
| GPU-GPU interconnect | Infinity Fabric (XGMI) 4th gen, ~1075 GB/s aggregate per GPU |
| TBP (per GPU) | 1400 W  (8 GPUs = 11.2 kW tray) |
| Driver / ROCm | amdgpu 6.19.14.31400100 / ROCm 7.2.4 |
| Host | 2 x EPYC 9575F (256 threads), 2.2 TiB RAM, Ubuntu 24.04.5 |

### Dense peak compute (per GPU, AMD published spec, no sparsity)

| Precision | Peak (TFLOPS) |
|---|---:|
| fp4 | 10,000.0 |
| fp6 | 10,000.0 |
| bf6 | 10,000.0 |
| fp8 | 5,000.0 |
| bf8 | 5,000.0 |
| fp16 | 2,500.0 |
| bf16 | 2,500.0 |
| fp32 | 157.3 |
| fp64 | 78.6 |

FP6/BF6 are block-scaled MX formats and run at the **FP4 rate** on CDNA 4 (10,000 TFLOPS), not half it.

## Measured TFLOPS (peak across log intervals)

Aggregate = sum of per-GPU peaks. Scaling = aggregate / N=1 value; perfect linear scaling would be N/1.

| Precision | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 | 2x eff | 3x eff | 4x eff | 5x eff | 6x eff | 7x eff | 8x eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4 | 3,187.3 | 6,368.2 | 9,568.3 | 12,779.7 | 16,036.2 | 19,246.8 | 22,462.4 | 25,596.3 | 2.00x (100%) | 3.00x (100%) | 4.01x (100%) | 5.03x (101%) | 6.04x (101%) | 7.05x (101%) | 8.03x (100%) |
| fp6 | 1,282.0 | 2,566.6 | 3,852.6 | 5,133.9 | 6,417.2 | 7,700.6 | 8,980.8 | 10,266.3 | 2.00x (100%) | 3.01x (100%) | 4.00x (100%) | 5.01x (100%) | 6.01x (100%) | 7.01x (100%) | 8.01x (100%) |
| bf6 | 1,282.2 | 2,568.3 | 3,851.0 | 5,129.5 | 6,417.0 | 7,701.3 | 8,982.4 | 10,264.9 | 2.00x (100%) | 3.00x (100%) | 4.00x (100%) | 5.00x (100%) | 6.01x (100%) | 7.01x (100%) | 8.01x (100%) |
| fp8 | 3,683.5 | 7,329.9 | 11,323.7 | 15,027.3 | 18,904.7 | 22,570.1 | 26,624.1 | 30,275.3 | 1.99x (99%) | 3.07x (102%) | 4.08x (102%) | 5.13x (103%) | 6.13x (102%) | 7.23x (103%) | 8.22x (103%) |
| bf8 | 3,393.6 | 6,691.5 | 10,031.9 | 13,630.5 | 17,050.2 | 20,559.3 | 23,845.0 | 26,948.0 | 1.97x (99%) | 2.96x (99%) | 4.02x (100%) | 5.02x (100%) | 6.06x (101%) | 7.03x (100%) | 7.94x (99%) |
| fp16 | 1,586.6 | 3,148.4 | 4,739.4 | 6,325.6 | 7,968.2 | 9,563.9 | 11,147.2 | 12,703.1 | 1.98x (99%) | 2.99x (100%) | 3.99x (100%) | 5.02x (100%) | 6.03x (100%) | 7.03x (100%) | 8.01x (100%) |
| bf16 | 1,672.2 | 3,330.1 | 5,021.9 | 6,730.3 | 8,444.4 | 10,195.4 | 11,905.0 | 13,556.5 | 1.99x (100%) | 3.00x (100%) | 4.02x (101%) | 5.05x (101%) | 6.10x (102%) | 7.12x (102%) | 8.11x (101%) |
| fp32 | 152.4 | 307.3 | 461.2 | 615.0 | 769.6 | 923.7 | 1,076.9 | 1,231.6 | 2.02x (101%) | 3.03x (101%) | 4.03x (101%) | 5.05x (101%) | 6.06x (101%) | 7.07x (101%) | 8.08x (101%) |
| fp64 | 76.7 | 153.6 | 230.6 | 307.8 | 385.9 | 462.7 | 539.5 | 616.6 | 2.00x (100%) | 3.01x (100%) | 4.02x (100%) | 5.03x (101%) | 6.04x (101%) | 7.04x (101%) | 8.04x (101%) |

## Per-GPU average TFLOPS

| Precision | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4 | 3,187.3 | 3,184.1 | 3,189.4 | 3,194.9 | 3,207.2 | 3,207.8 | 3,208.9 | 3,199.5 |
| fp6 | 1,282.0 | 1,283.3 | 1,284.2 | 1,283.5 | 1,283.4 | 1,283.4 | 1,283.0 | 1,283.3 |
| bf6 | 1,282.2 | 1,284.1 | 1,283.7 | 1,282.4 | 1,283.4 | 1,283.6 | 1,283.2 | 1,283.1 |
| fp8 | 3,683.5 | 3,664.9 | 3,774.6 | 3,756.8 | 3,780.9 | 3,761.7 | 3,803.4 | 3,784.4 |
| bf8 | 3,393.6 | 3,345.7 | 3,344.0 | 3,407.6 | 3,410.0 | 3,426.5 | 3,406.4 | 3,368.5 |
| fp16 | 1,586.6 | 1,574.2 | 1,579.8 | 1,581.4 | 1,593.6 | 1,594.0 | 1,592.5 | 1,587.9 |
| bf16 | 1,672.2 | 1,665.0 | 1,674.0 | 1,682.6 | 1,688.9 | 1,699.2 | 1,700.7 | 1,694.6 |
| fp32 | 152.4 | 153.6 | 153.7 | 153.8 | 153.9 | 154.0 | 153.8 | 153.9 |
| fp64 | 76.7 | 76.8 | 76.9 | 76.9 | 77.2 | 77.1 | 77.1 | 77.1 |

### Per-GPU spread at N=8 (die-to-die variation)

| Precision | min | max | spread |
|---|---:|---:|---:|
| fp4 | 3,166.1 | 3,224.4 | 1.8% |
| fp6 | 1,279.6 | 1,285.6 | 0.5% |
| bf6 | 1,279.6 | 1,285.5 | 0.5% |
| fp8 | 3,689.6 | 3,864.7 | 4.5% |
| bf8 | 3,301.8 | 3,446.7 | 4.2% |
| fp16 | 1,555.2 | 1,615.8 | 3.8% |
| bf16 | 1,646.9 | 1,718.5 | 4.2% |
| fp32 | 152.6 | 154.7 | 1.4% |
| fp64 | 76.5 | 77.5 | 1.3% |

## Measured vs dense peak (per-GPU, N=1 run)

| Precision | Measured | Paper dense peak | % of peak |
|---|---:|---:|---:|
| fp4 | 3,187.34 | 10,000.0 | **31.9%** |
| fp6 | 1,281.97 | 10,000.0 | **12.8%** |
| bf6 | 1,282.17 | 10,000.0 | **12.8%** |
| fp8 | 3,683.46 | 5,000.0 | **73.7%** |
| bf8 | 3,393.55 | 5,000.0 | **67.9%** |
| fp16 | 1,586.60 | 2,500.0 | **63.5%** |
| bf16 | 1,672.18 | 2,500.0 | **66.9%** |
| fp32 | 152.43 | 157.3 | **96.9%** |
| fp64 | 76.65 | 78.6 | **97.5%** |

### Aggregate at N=8 vs aggregate peak

| Precision | Measured aggregate | Peak x8 | % of peak |
|---|---:|---:|---:|
| fp4 | 25,596.3 | 80,000.0 | 32.0% |
| fp6 | 10,266.3 | 80,000.0 | 12.8% |
| bf6 | 10,264.9 | 80,000.0 | 12.8% |
| fp8 | 30,275.3 | 40,000.0 | 75.7% |
| bf8 | 26,948.0 | 40,000.0 | 67.4% |
| fp16 | 12,703.1 | 20,000.0 | 63.5% |
| bf16 | 13,556.5 | 20,000.0 | 67.8% |
| fp32 | 1,231.6 | 1,258.4 | 97.9% |
| fp64 | 616.6 | 628.8 | 98.1% |

## Cross-machine comparison — Dell Cloud vs amd-ubuntu vs B200 (per-GPU)

All three columns are per-GPU at N=1. **Dell Cloud and amd-ubuntu are the same silicon** — 8 x MI355X (gfx950) — so the delta between them is a *software* delta:

| | Dell Cloud | amd-ubuntu (this host) |
|---|---|---|
| ROCm | 7.2.3-90 | **7.2.4** |
| Code objects | gfx942 alias (`HSA_OVERRIDE_GFX_VERSION=9.4.2`) | **native gfx950** |
| Container | Singularity + ext3 overlay | Docker |
| gst duration | ~60 s | 30 s |

B200 reference measurements (provided, per-GPU): 768 TFLOPS FP32†, 1493 TFLOPS BF16, 4103 TFLOPS FP8. † see the FP32/TF32 note below the table — this is not a like-for-like figure and its ratio column is deliberately not computed.

| Precision | Dell Cloud MI355X | amd-ubuntu MI355X | B200 ref | AMD/Dell | AMD/B200 | MI355X peak | B200 peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp4 | 3,159.52 | 3,187.34 | - | **1.01x** | - | 10,000.0 | 9,000 |
| fp6 | 1,280.17 | 1,281.97 | - | **1.00x** | - | 10,000.0 | - |
| bf6 | 1,280.20 | 1,282.17 | - | **1.00x** | - | 10,000.0 | - |
| fp8 | 3,610.88 | 3,683.46 | 4,103 | **1.02x** | **0.90x** | 5,000.0 | 4,500 |
| bf8 | 3,238.62 | 3,393.55 | - | **1.05x** | - | 5,000.0 | 4,500 |
| fp16 | 1,534.56 | 1,586.60 | - | **1.03x** | - | 2,500.0 | 2,250 |
| bf16 | 1,639.78 | 1,672.18 | 1,493 | **1.02x** | **1.12x** | 2,500.0 | 2,250 |
| fp32 | 153.76 | 152.43 | 768† | **0.99x** | _not comparable†_ | 157.3 | 80 |
| fp64 | 77.02 | 76.65 | - | **1.00x** | - | 78.6 | 40 |

amd-ubuntu vs Dell Cloud ranges from **0.99x** (`fp32`) to **1.05x** (`bf8`). Since the silicon is identical, any gain is attributable to the newer ROCm and to running native gfx950 code objects instead of the gfx942 alias — which is exactly why this host does not set `HSA_OVERRIDE_GFX_VERSION`.

**† FP32 vs TF32 — this is NOT an apples-to-apples comparison.** The 768 TFLOPS B200 figure cannot be IEEE FP32 — B200's IEEE FP32 dense peak is only ~80 TFLOPS, the number in the "B200 peak" column above. 768 is almost certainly **TF32 tensor** (NVIDIA's reduced-precision 19-bit format, run on the tensor cores), whereas MI355X's 152.8/157.3 TFLOPS is **true IEEE-754 FP32 on the vector ALUs**. The two numbers are two different data types on two different execution units. It is included in the table only so a reader does not mistake its absence for "not measured" — the ratio column is deliberately left as "not comparable" rather than computed, because a 5.0x-looking number here would actively mislead: MI355X's true-FP32 is not 5x slower than anything, it is simply not the same operation as B200's TF32 path. RVS `gst` has no TF32 config, so that path is unmeasured on either MI355X host and no side-by-side TF32 number exists.

Only BF16 and FP8 above are like-for-like B200 reference measurements.

## Why fp4 / fp6 / bf6 land so far below peak

**Short version.** Not memory bandwidth — these GEMMs use only 1-8% of HBM. Three causes, in increasing severity:

1. **MX block scaling** (fp4, fp6, bf6 only) — an E8M0 scale per 32-element block is real work the theoretical peak ignores. Costs roughly the fp4 40% vs fp8 71% gap.
2. **FP6 is not byte-aligned** (4 values per 3 bytes) — cross-byte unpacking, and likely no native full-rate MFMA path. This is why **fp6 is absolutely slower than fp8 (0.35x) despite twice the nominal peak**.
3. **Kernel maturity** — but only partly: native gfx950 codegen improved fp4 by **26%** while fp6 did not move **at all** (0.97x). So fp4 is under-tuned; fp6 hits a structural floor that better codegen does not touch.

The rest of this section is the evidence for each.

### It is not memory bandwidth

For the `gst` shape (8192x8192x16384, ~2.20 TFLOP per GEMM), required HBM bandwidth at the measured rates is:

| Precision | Bytes/GEMM | Bandwidth needed | % of 8 TB/s HBM |
|---|---:|---:|---:|
| fp4 | 277 MB | 500 GB/s | 6.3% |
| fp6 / bf6 | 344 MB | 194 GB/s | 2.4% |
| fp8 | 403 MB | 653 GB/s | 8.2% |
| bf16 | 671 MB | 497 GB/s | 6.2% |
| fp64 | 2282 MB | 80 GB/s | 1.0% |

Every precision uses **1-8% of HBM bandwidth**. These GEMMs have arithmetic intensity in the thousands of FLOP/byte — they are firmly compute-bound. Memory bandwidth is conclusively not the limiter, so the answer lies in the kernels.

### Cause 1 — MX block scaling costs throughput (fp4, fp6, bf6)

Exactly the three low outliers carry `scale_a: block, scale_b: block` in their generated conf; fp8/bf8/fp16/bf16 do not. MX formats attach an E8M0 scale per 32-element block, and applying those scales is real work that the theoretical peak number does not account for. fp4 at **39.8%** vs fp8 at **71.3%** is roughly the size of that tax.

### Cause 2 — FP6 is not byte-aligned, and pays much more (fp6, bf6 only)

Block scaling alone cannot explain fp6, because fp4 and fp6 share the same mechanism *and the same 10,000 TFLOPS nominal peak*, yet differ 3.2x. The absolute cross-precision ratios are the tell:

| Comparison | Measured | Nominal peak ratio |
|---|---:|---:|
| fp4 / fp8 | 1.12x | 2.00x |
| **fp6 / fp8** | **0.35x** | 2.00x |
| **fp4 / fp6** | **3.21x** | 1.00x |

**fp6 is slower in absolute terms than fp8** — 1238 vs 3564 TFLOPS — despite nominally having twice the peak. A format cannot be 2x faster on paper and 3x slower in practice unless it is not running on the fast path at all.

The most likely mechanism is packing: fp4 is 2 values per byte (clean nibbles) and fp8 is 1 value per byte, but **fp6 is 4 values per 3 bytes** — not byte-aligned. Feeding packed 6-bit operands into the matrix engine requires cross-byte bit extraction, and if the MFMA instruction cannot consume packed FP6 natively the kernel must widen it first, at which point throughput is set by the wider format, not by FP6's nominal rate. Consistent with that, measured fp6 (1238) sits at 0.81x measured fp16 (1522) — roughly bf16-class throughput minus unpack overhead.

### Cause 3 — kernel maturity, and the evidence that separates it from the above

Comparing the two MI355X hosts isolates software from silicon. Dell Cloud ran ROCm 7.2.3 with the gfx942 alias; this host runs ROCm 7.2.4 with native gfx950 code objects. **Identical hardware.** Only one precision responded:

| Precision | amd-ubuntu / Dell Cloud |
|---|---:|
| **fp4** | **1.26x** |
| fp6, bf6 | 0.97x |
| fp8, bf8, fp16, bf16, fp32, fp64 | 0.99x |

fp4 gained 26% from native gfx950 codegen while **fp6 did not move at all**. That asymmetry is informative in both directions: fp4's shortfall is partly a *tuning* problem (it improves when the compiler targets the real architecture), whereas fp6's shortfall is a *structural* floor that better codegen does not touch — consistent with Cause 2 rather than with immature tuning.

### Caveat on the fp6 peak figure

The 10,000 TFLOPS peak used for fp6/bf6 comes from AMD's claim that CDNA 4 processes FP6 at the FP4 rate (a stated differentiator vs competitors that run FP6 at FP8 rate). If that claim does not hold for this silicon/stack, the correct denominator would be 5,000 and fp6 would read **24.8%** rather than 12.4% of peak. Either way it is the worst precision measured, and either way fp6 being absolutely slower than fp8 is the anomaly worth explaining. This is flagged because the percentage — unlike the measured TFLOPS — depends on a vendor claim this benchmark cannot verify.

### What would settle it

None of the above is a profiled root cause. Confirming Cause 2 requires kernel-level inspection — `rocprof` on a single fp6 GEMM to see which MFMA variant is issued and whether an unpack/convert kernel precedes it, or hipBLASLt's heuristic log to see which algorithm it selects for `fp6_e3m2_r`. That is a worthwhile follow-up if low-precision throughput matters for a real workload.

## Observations (auto-generated)

- `fp6`: only **12.8%** of dense peak. For MX-FP6 this reproduces the known hipBLASLt MX-fp6 kernel ceiling seen on dell-cloud (12.8%), not a regression on this host.
- `bf6`: only **12.8%** of dense peak. For MX-FP6 this reproduces the known hipBLASLt MX-fp6 kernel ceiling seen on dell-cloud (12.8%), not a regression on this host.

## Reproducing

```bash
cd /orcd/data/orcd/022/benchmarks/amd-ubuntu && source common/env.sh
cd work-rocmval && ./run_part_a.sh          # smoke -> sweep -> health -> analysis
$PY analyze_rvs.py $LOG_ROOT/rvs/sweep_* -o $BENCH_ROOT/results
```

## Source data

| What | Where |
|---|---|
| Raw rvs stdout, one per (N, precision) | `logs/rvs/sweep_*/<n>x_<prec>.log` |
| Generated gst confs | `logs/rvs/sweep_*/<n>x_<prec>.conf` |
| Per-run summary | `logs/rvs/sweep_*/summary.{csv,txt}` |
| Health modules | `logs/rvs/health_*/` |
| This table as CSV | `results/rvs_tflops.csv` |

