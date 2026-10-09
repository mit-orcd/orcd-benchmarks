# Kimi-K3 — expert parallelism vs TP-only, matched admission cap

Both arms: MAD-pinned image, MAD env vars, `--max-num-seqs 256`, TP=8, ISL/OSL 1024/1024.
The **only** difference between them is `--enable-expert-parallel`.

This supersedes the A/B in `kimi-k3-mad.md` section 6, where the two arms accidentally used
different `--max-num-seqs` (64 vs 256) and only the c=64 point was comparable. Here the cap
is identical, so every concurrency point is a valid comparison.

---

<!-- BEGIN kimi-ep-ab (auto-generated) -->

## 6. Experiment — expert parallelism (EP) on vs off

Section 5 flagged EP as the top tuning lever: with EP off, TP shards every expert across all 8 GPUs so each GPU reads a slice of *every* activated expert (~116 GB/step, ~29% of HBM bandwidth), while XGMI sits ~1% utilized. EP places whole experts on specific ranks — fewer, complete reads per GPU, paid for with all-to-all token routing over the idle interconnect.

Both runs use the identical validated recipe flag set; the **only** difference is `--enable-expert-parallel`. Baseline is the reference result (EP is not part of the validated recipe).

- Baseline (TP-only): `tp_only`
- EP enabled: `ep`

TPOT = time per output token.

TTFT = time to first token.

| Concurrency | tok/s TP-only | tok/s EP | EP/TP | TPOT TP-only (ms) | TPOT EP (ms) | TTFT TP-only (ms) | TTFT EP (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1,212.5 | 996.1 | **0.82x** | 51.74 | 56.19 | 261.4 | 271.1 |
| 128 | 1,773.3 | 1,657.2 | 0.93x | 71.35 | 76.22 | 357.3 | 356.1 |
| 256 | 2,543.2 | 2,393.9 | 0.94x | 100.61 | 106.21 | 425.1 | 473.9 |

**Headline (c=256, the batch where weight traffic dominates): 0.94x**  ·  range 0.82x (c=64) to 0.94x (c=256).

**Roughly neutral (0.94x at c=256).** The HBM saving and the added all-to-all latency approximately cancel. That is itself informative: it means the workload is not *purely* bandwidth-bound the way section 3's utilization figures imply, and that all-to-all round-trip latency across 92 MoE layers is a first-order cost, not a rounding error.

**Caveat on interpretation.** Bandwidth *utilization* (29% HBM vs 1% XGMI) motivated this experiment, but utilization measures throughput headroom, not latency exposure. A collective can be far from bandwidth-saturated and still dominate step time if it serializes. Whatever the sign of the result above, the honest lesson is that the two resources are not interchangeable just because one has spare capacity.

Raw data: per-concurrency JSON in the two sweep directories listed above.

<!-- END kimi-ep-ab -->

---

## Source data

| What | Where |
|---|---|
| TP-only arm | `kimi_ep_matched_20261002_122247/tp_only/c<N>.{json,log}` |
| EP arm | `kimi_ep_matched_20261002_122247/ep/c<N>.{json,log}` |
| Driver state | `kimi_ep_matched_20261002_122247/STATE.txt` |
