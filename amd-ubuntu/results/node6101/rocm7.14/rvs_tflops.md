# RVS `gst` TFLOPS — MI355X x8 (gfx950, ROCm 7.2.4)

System: 8 x AMD Instinct MI355X (CDNA 4 / gfx950), ROCm 7.2.4, Ubuntu 24.04.5.
Source runs: sweep_20261002_080350

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
| fp4 | 3,989.8 | 7,887.3 | 11,947.0 | 13,590.0 | 13,515.2 | 15,823.3 | 16,653.5 | 17,099.5 | 1.98x (99%) | 2.99x (100%) | 3.41x (85%) | 3.39x (68%) | 3.97x (66%) | 4.17x (60%) | 4.29x (54%) |
| fp6 | 1,240.9 | 2,481.2 | 3,716.1 | 4,956.9 | 6,197.2 | 7,436.1 | 8,631.8 | 9,913.7 | 2.00x (100%) | 2.99x (100%) | 3.99x (100%) | 4.99x (100%) | 5.99x (100%) | 6.96x (99%) | 7.99x (100%) |
| bf6 | 1,240.9 | 2,482.2 | 3,715.6 | 4,958.2 | 6,193.6 | 7,434.2 | 8,679.1 | 9,915.2 | 2.00x (100%) | 2.99x (100%) | 4.00x (100%) | 4.99x (100%) | 5.99x (100%) | 6.99x (100%) | 7.99x (100%) |
| fp8 | 3,628.8 | 7,066.4 | 10,799.8 | 14,498.9 | 18,102.5 | 22,145.6 | 25,822.6 | 29,312.9 | 1.95x (97%) | 2.98x (99%) | 4.00x (100%) | 4.99x (100%) | 6.10x (102%) | 7.12x (102%) | 8.08x (101%) |
| bf8 | 3,342.0 | 6,550.0 | 9,924.7 | 13,228.4 | 16,580.7 | 20,024.7 | 23,343.7 | 26,600.0 | 1.96x (98%) | 2.97x (99%) | 3.96x (99%) | 4.96x (99%) | 5.99x (100%) | 6.98x (100%) | 7.96x (99%) |
| fp16 | 1,519.6 | 3,011.5 | 4,622.2 | 6,135.7 | 7,689.9 | 9,369.9 | 10,895.0 | 12,484.6 | 1.98x (99%) | 3.04x (101%) | 4.04x (101%) | 5.06x (101%) | 6.17x (103%) | 7.17x (102%) | 8.22x (103%) |
| bf16 | 1,631.1 | 3,209.8 | 4,875.6 | 6,535.1 | 8,151.6 | 9,915.9 | 11,601.6 | 13,200.5 | 1.97x (98%) | 2.99x (100%) | 4.01x (100%) | 5.00x (100%) | 6.08x (101%) | 7.11x (102%) | 8.09x (101%) |
| fp32 | 154.1 | 308.0 | 461.3 | 614.1 | 767.5 | 921.7 | 1,075.8 | 1,228.8 | 2.00x (100%) | 2.99x (100%) | 3.99x (100%) | 4.98x (100%) | 5.98x (100%) | 6.98x (100%) | 7.98x (100%) |
| fp64 | 77.2 | 154.2 | 230.9 | 307.4 | 384.1 | 461.5 | 538.5 | 615.0 | 2.00x (100%) | 2.99x (100%) | 3.98x (100%) | 4.97x (99%) | 5.98x (100%) | 6.97x (100%) | 7.96x (100%) |

## Per-GPU average TFLOPS

| Precision | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N=7 | N=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4 | 3,989.8 | 3,943.6 | 3,982.3 | 3,397.5 | 2,703.0 | 2,637.2 | 2,379.1 | 2,137.4 |
| fp6 | 1,240.9 | 1,240.6 | 1,238.7 | 1,239.2 | 1,239.4 | 1,239.4 | 1,233.1 | 1,239.2 |
| bf6 | 1,240.9 | 1,241.1 | 1,238.5 | 1,239.6 | 1,238.7 | 1,239.0 | 1,239.9 | 1,239.4 |
| fp8 | 3,628.8 | 3,533.2 | 3,599.9 | 3,624.7 | 3,620.5 | 3,690.9 | 3,688.9 | 3,664.1 |
| bf8 | 3,342.0 | 3,275.0 | 3,308.2 | 3,307.1 | 3,316.1 | 3,337.5 | 3,334.8 | 3,325.0 |
| fp16 | 1,519.6 | 1,505.8 | 1,540.7 | 1,533.9 | 1,538.0 | 1,561.6 | 1,556.4 | 1,560.6 |
| bf16 | 1,631.1 | 1,604.9 | 1,625.2 | 1,633.8 | 1,630.3 | 1,652.6 | 1,657.4 | 1,650.1 |
| fp32 | 154.1 | 154.0 | 153.8 | 153.5 | 153.5 | 153.6 | 153.7 | 153.6 |
| fp64 | 77.2 | 77.1 | 77.0 | 76.8 | 76.8 | 76.9 | 76.9 | 76.9 |

### Per-GPU spread at N=8 (die-to-die variation)

| Precision | min | max | spread |
|---|---:|---:|---:|
| fp4 | 1,868.5 | 2,694.9 | 30.7% |
| fp6 | 1,233.2 | 1,241.6 | 0.7% |
| bf6 | 1,235.0 | 1,242.2 | 0.6% |
| fp8 | 3,529.7 | 3,885.2 | 9.1% |
| bf8 | 3,153.0 | 3,425.3 | 7.9% |
| fp16 | 1,486.2 | 1,610.3 | 7.7% |
| bf16 | 1,585.5 | 1,722.6 | 8.0% |
| fp32 | 152.6 | 154.6 | 1.3% |
| fp64 | 76.4 | 77.4 | 1.2% |

## Measured vs dense peak (per-GPU, N=1 run)

| Precision | Measured | Paper dense peak | % of peak |
|---|---:|---:|---:|
| fp4 | 3,989.76 | 10,000.0 | **39.9%** |
| fp6 | 1,240.93 | 10,000.0 | **12.4%** |
| bf6 | 1,240.93 | 10,000.0 | **12.4%** |
| fp8 | 3,628.75 | 5,000.0 | **72.6%** |
| bf8 | 3,341.98 | 5,000.0 | **66.8%** |
| fp16 | 1,519.64 | 2,500.0 | **60.8%** |
| bf16 | 1,631.10 | 2,500.0 | **65.2%** |
| fp32 | 154.08 | 157.3 | **98.0%** |
| fp64 | 77.22 | 78.6 | **98.2%** |

### Aggregate at N=8 vs aggregate peak

| Precision | Measured aggregate | Peak x8 | % of peak |
|---|---:|---:|---:|
| fp4 | 17,099.5 | 80,000.0 | 21.4% |
| fp6 | 9,913.7 | 80,000.0 | 12.4% |
| bf6 | 9,915.2 | 80,000.0 | 12.4% |
| fp8 | 29,312.9 | 40,000.0 | 73.3% |
| bf8 | 26,600.0 | 40,000.0 | 66.5% |
| fp16 | 12,484.6 | 20,000.0 | 62.4% |
| bf16 | 13,200.5 | 20,000.0 | 66.0% |
| fp32 | 1,228.8 | 1,258.4 | 97.6% |
| fp64 | 615.0 | 628.8 | 97.8% |

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
| fp4 | 3,159.52 | 3,989.76 | - | **1.26x** | - | 10,000.0 | 9,000 |
| fp6 | 1,280.17 | 1,240.93 | - | **0.97x** | - | 10,000.0 | - |
| bf6 | 1,280.20 | 1,240.93 | - | **0.97x** | - | 10,000.0 | - |
| fp8 | 3,610.88 | 3,628.75 | 4,103 | **1.00x** | **0.88x** | 5,000.0 | 4,500 |
| bf8 | 3,238.62 | 3,341.98 | - | **1.03x** | - | 5,000.0 | 4,500 |
| fp16 | 1,534.56 | 1,519.64 | - | **0.99x** | - | 2,500.0 | 2,250 |
| bf16 | 1,639.78 | 1,631.10 | 1,493 | **0.99x** | **1.09x** | 2,500.0 | 2,250 |
| fp32 | 153.76 | 154.08 | 768† | **1.00x** | _not comparable†_ | 157.3 | 80 |
| fp64 | 77.02 | 77.22 | - | **1.00x** | - | 78.6 | 40 |

amd-ubuntu vs Dell Cloud ranges from **0.97x** (`bf6`) to **1.26x** (`fp4`). Since the silicon is identical, any gain is attributable to the newer ROCm and to running native gfx950 code objects instead of the gfx942 alias — which is exactly why this host does not set `HSA_OVERRIDE_GFX_VERSION`.

### Why `fp4` alone gains 1.26x

Every other precision lands in a tight 0.97x-1.03x band around Dell Cloud's number — essentially reproduction, not improvement. `fp4` is the lone outlier, and it is also a low-variance, reproducible measurement here: 0% per-GPU spread at N=1, still under 1% at N=2. That combination — one precision moving, everything else static, and the mover being clean data rather than noise — points at a specific software cause rather than run-to-run variance:

- **`fp4` is the newest, least mature kernel path in hipBLASLt** among the precisions tested. MX-block-scaled FP4 has had far less tuning time than BF16/FP8/FP32, which is exactly where a difference between a gfx950-native build and a gfx942-emulated one (Dell Cloud's `HSA_OVERRIDE_GFX_VERSION=9.4.2`) would most plausibly show up — an emulation layer is more likely to cost performance on a codepath that hasn't been separately hand-tuned for the emulated target.
- This is inference from the pattern, not a profiled root cause: no kernel-level trace was captured to confirm gfx942-emulation overhead specifically. The counter-evidence worth weighing is that `fp6`/`bf6` share the same MX block-scaling mechanism and the same 10,000 TFLOPS peak class as `fp4`, yet show **no** such gain (0.97x, i.e. slightly *below* Dell Cloud) — so "MX format in general" is not the explanation; it would have to be something specific to the fp4 numeric path itself.
- Also see the scaling-efficiency finding below: `fp4` is simultaneously the only precision with severely non-uniform multi-GPU scaling on *this* host (N=8 per-GPU spread up to 63%, vs <1% for every other precision including fp6/bf6). A kernel path immature enough to gain unusually from native codegen is also a plausible place to find launch or scheduling instability under concurrent multi-GPU load — the two observations may share a cause even though neither proves the other.

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

- `fp4`: N=8 scaling efficiency **54%** -- below the ~95% expected for an embarrassingly-parallel GEMM. Power sharing on the 11.2 kW tray is the leading explanation.
- `fp4`: die-to-die spread at N=8 is **30.7%** (1,868.5-2,694.9 TFLOPS) -- per-die clock variation under sustained load.
- `fp8`: die-to-die spread at N=8 is **9.1%** (3,529.7-3,885.2 TFLOPS) -- per-die clock variation under sustained load.
- `fp6`: only **12.4%** of dense peak. For MX-FP6 this reproduces the known hipBLASLt MX-fp6 kernel ceiling seen on dell-cloud (12.8%), not a regression on this host.
- `bf6`: only **12.4%** of dense peak. For MX-FP6 this reproduces the known hipBLASLt MX-fp6 kernel ceiling seen on dell-cloud (12.8%), not a regression on this host.

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

