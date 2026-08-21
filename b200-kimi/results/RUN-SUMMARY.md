# Kimi-K3 on B200 — run summary

Generated 2026-08-21T14:30:59-04:00 by job `regen`.

Chain: `gate:20916740 verify:20916741 base:20916742 summary:20916743`

## Stage outcomes

| Stage | Job | State | Elapsed |
|---|---:|---|---:|
| gate | 20916740 | COMPLETED | 00:00:32 |
| verify | 20916741 | COMPLETED | 00:01:39 |
| base | 20916742 | COMPLETED | 00:40:32 |
| summary | 20916743 | COMPLETED | 00:00:03 |

## Hardware and staging (from the probe)

```
===== node5500-c1 =====
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
MODEL VISIBLE: /orcd/compute/orcd/025/models/Kimi-K3
  96/96 shards, 1560936091448 bytes
DRIVER OK: 590.48.01 (r590 >= r580) -- the cu130 vllm:kimi-k3 image can run here
===== node5501-c1 =====
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
MODEL VISIBLE: /orcd/compute/orcd/025/models/Kimi-K3
  96/96 shards, 1560936091448 bytes
DRIVER OK: 590.48.01 (r590 >= r580) -- the cu130 vllm:kimi-k3 image can run here
===== node5700-c1 =====
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
MODEL VISIBLE: /orcd/compute/orcd/025/models/Kimi-K3
  96/96 shards, 1560936091448 bytes
DRIVER OK: 590.48.01 (r590 >= r580) -- the cu130 vllm:kimi-k3 image can run here
===== node5701-c1 =====
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
MODEL VISIBLE: /orcd/compute/orcd/025/models/Kimi-K3
  96/96 shards, 1560936091448 bytes
DRIVER OK: 590.48.01 (r590 >= r580) -- the cu130 vllm:kimi-k3 image can run here
```

## Pre-flight verification (2 nodes, no weight load)

Run dir: `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/verify_20260821_125833`

```
[2026-08-21T12:58:33-04:00] --- check 1: routable IPv4 for the head node ---
[2026-08-21T12:58:33-04:00] resolve_ipv4(node5700-c1) = 10.1.57.64
[2026-08-21T12:58:33-04:00] OK: head IP is routable IPv4
[2026-08-21T12:58:33-04:00] --- check 2: vLLM can inspect KimiK3 (triggers the Triton compile) ---
[2026-08-21T12:59:33-04:00] OK: model inspection succeeded (Triton compile worked)
[2026-08-21T12:59:33-04:00] --- check 2b: filelock works on the redirected JIT cache dirs ---
[2026-08-21T12:59:49-04:00] OK: flock works on every JIT cache dir
[2026-08-21T12:59:49-04:00] --- check 3: ray forms a 16-GPU cluster across both nodes ---
[2026-08-21T12:59:59-04:00] ray cluster GPUs: 16/16
[2026-08-21T12:59:59-04:00] OK: 2-node ray cluster formed
[2026-08-21T13:00:10-04:00] VERIFY PASSED -- both prior failure modes are fixed
```

## Known failure modes already fixed

| Symptom | Root cause | Fix |
|---|---|---|
| `FATAL: "python": executable file not found` | image has only `python3` | `$PY_C=python3` everywhere |
| `ModuleNotFoundError: ray` | ray is not in the vLLM image | ray 2.57.0 installed `--no-deps` into `pylibs/` |
| `transformers` pulling in a broken TensorFlow | host `~/.local` site-packages shadowed the container's | `PYTHONNOUSERSITE=1` + explicit `PYTHONPATH` |
| `bits/libc-header-start.h: No such file` during model inspection | host Spack gcc leaked via `PATH`; Triton JIT used it against container headers | `PATH`/`CC`/`CXX` pinned to the container |
| ray GCS unreachable at `[fe80::...]:6379` | `getent hosts` returned an IPv6 link-local on the compute node | `resolve_ipv4()` forces a routable IPv4 |
| 2-node job launched behind a failed gate | a CANCELLED job satisfies `afterany` | `base` now needs `afterok:verify` too |

## Single-node attempt (TP8 × PP1)

Run dir: `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/kimi_1node_20260820_163610`

```
[2026-08-20T16:36:13-04:00] parallelism OK: TP=8 within each node x PP=1 across 1 node(s) = 8 GPUs
[2026-08-20T16:36:13-04:00] model OK (pre-staged, read-only): /orcd/compute/orcd/025/models/Kimi-K3
[2026-08-20T16:36:15-04:00] per-GPU HBM: 183359 MiB; node total ~1538 GB vs 1561 GB of weights
[2026-08-20T16:39:39-04:00] VERDICT: out of memory during model load.
```

**Result: the checkpoint does not fit on one B200 node.** This is the finding the
single-node stage exists to establish, not a failure to fix.

`torch.OutOfMemoryError` was raised on **8 of 8 GPUs**, at model load:

```
CUDA out of memory. Tried to allocate 588.00 MiB. GPU 4 has a total capacity of 178.35 GiB of which 108.00 MiB is free.
```

| | Per GPU | × 8 (node) |
|---|---:|---:|
| Usable HBM | 178.35 GiB | 1426.8 GiB |
| Occupied before the failing allocation | 178.24 GiB | 1425.9 GiB |
| Kimi-K3 weights | — | 1454.2 GiB (1561 GB) |

Every GPU was filled to within ~0.1 GiB of capacity and the model still did not
fit, which is why the benchmark runs TP8 × PP2 across two nodes.

## Two-node run (TP8 × PP2)

Run dir: `/orcd/data/orcd/022/benchmarks/b200-kimi/logs/kimi_base_20260821_130024`

```
[2026-08-21T13:00:25-04:00] parallelism OK: TP=8 within each node x PP=2 across 2 node(s) = 16 GPUs
[2026-08-21T13:00:25-04:00] image OK: /orcd/data/orcd/022/benchmarks/b200-kimi/imag/vllm-openai_kimi-k3.sif (7921852416 bytes, manifest verified)
[2026-08-21T13:00:26-04:00] model OK (pre-staged, read-only): /orcd/compute/orcd/025/models/Kimi-K3
[2026-08-21T13:11:05-04:00] server up after 634s
[2026-08-21T13:40:33-04:00] sweep rc=0
[2026-08-21T13:40:50-04:00] analyze rc=0 -> /orcd/data/orcd/022/benchmarks/b200-kimi/results/kimi-k3-base-b200.md
[2026-08-21T13:40:51-04:00] DONE. run dir: /orcd/data/orcd/022/benchmarks/b200-kimi/logs/kimi_base_20260821_130024
```

### Sweep

```
vLLM serving sweep 2026-08-21T13:11:07-04:00
model    : /orcd/compute/orcd/025/models/Kimi-K3
served   : Kimi-K3
host:port: node5700-c1:8000
TP/PP    : 8/2    max_num_seqs=64 max_model_len=16384
ISL/OSL  : 1024/1024  range_ratio=0.8  seed=42
conc     : 1 2 4 8 16 32 64
image    : /orcd/data/orcd/022/benchmarks/b200-kimi/imag/vllm-openai_kimi-k3.sif

  conc        req/s    out_tok/s  ttft_ms_med  tpot_ms_med
     1         0.08         86.7        225.9        11.24   rc=0 166s completed=10
     2         0.16        165.2        217.6        11.85   rc=0 161s completed=20
     4         0.26        280.8        233.3        13.62   rc=0 195s completed=40
     8         0.44        475.3        234.5        15.41   rc=0 227s completed=80
    16         0.77        801.8        234.6        18.65   rc=0 253s completed=160
    32         1.16       1209.3        264.4        25.28   rc=0 320s completed=320
    64         1.64       1696.4        278.1        35.89   rc=0 442s completed=640
results: /orcd/data/orcd/022/benchmarks/b200-kimi/logs/kimi_base_20260821_130024/sweep
```

## Key finding — the bottleneck

**LATENCY-BOUND, NOT BANDWIDTH-BOUND.**

HBM sits at only **~23% of peak** — the bandwidth is there and is going
unused. What binds is memory-level parallelism, not memory speed: at peak
concurrency the routed experts are spread so thin that each one sees only
**1.7 tokens**, making every expert GEMM a matrix-*vector* product. A GEMV
cannot keep enough concurrent memory requests in flight to saturate HBM.

Weight *traffic volume* dominates step time, but the ceiling being hit is
memory-access latency/occupancy. Calling this "HBM-bandwidth-bound" would point at
the wrong fix: faster memory would buy nothing. Widening the GEMMs buys everything
— raise `--max-num-seqs` (64 -> 256+), or use speculative decoding. Weight bytes
plateau above batch ~512, so beyond that extra tokens are nearly free.

See section 3 of `kimi-k3-base-b200.md` for the full derivation.

## Reports

- `/orcd/data/orcd/022/benchmarks/b200-kimi/results/kimi-k3-base-b200.md` (400 lines, 2026-08-21T14:30:25-04:00)
- `/orcd/data/orcd/022/benchmarks/b200-kimi/results/kimi-k3-base-b200.csv` (15 lines, 2026-08-21T14:30:25-04:00)

## Comparison baseline

MI355X: `/orcd/data/orcd/022/benchmarks/amd-benchmarks/amd-cloud/results/kimi-k3-base.md`, measured 2026-08-14 (ATOM, TP8, 1 node).
