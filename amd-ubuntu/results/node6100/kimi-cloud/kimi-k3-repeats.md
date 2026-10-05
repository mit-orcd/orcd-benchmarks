# Kimi-K3 — repeatability of the MAD-vs-original gap

`kimi-k3-comparison.md` reported the MAD recipe at **0.91× the original** from a
single run each, and flagged that one run per config cannot support a claim that
size. This run repeats c=64 on both configs to separate signal from noise.

Source: `logs/atom/kimi_repeats_20261002_112609/`

## Per-repeat measurements

| Config | Rep | tok/s | TTFT med (ms) | TPOT med (ms) | completed |
|---|---:|---:|---:|---:|---:|
| `A_original` | 1 | 1,387.0 | 228.7 | 44.83 | 640 |
| `A_original` | 2 | 1,413.2 | 225.7 | 44.40 | 640 |
| `A_original` | 3 | 1,411.7 | 224.2 | 44.36 | 640 |
| `B_mad` | 1 | 1,210.3 | 262.5 | 51.98 | 640 |
| `B_mad` | 2 | 1,212.9 | 262.4 | 51.87 | 640 |
| `B_mad` | 3 | 1,212.2 | 262.6 | 51.88 | 640 |

## Statistics

| Config | n | mean tok/s | stdev | spread (max−min) | rel. spread |
|---|---:|---:|---:|---:|---:|
| `A_original` | 3 | **1,403.9** | 14.7 | 26.2 | 1.9% |
| `B_mad` | 3 | **1,211.8** | 1.4 | 2.6 | 0.2% |

## Verdict

- Original (`A_original`): **1,403.9 ± 14.7** tok/s (n=3)
- MAD (`B_mad`): **1,211.8 ± 1.4** tok/s (n=3)
- **Ratio MAD/original = 0.863×** (single-run estimate was 0.908×)

**The gap is real.** The 192.2 tok/s difference between configs is larger than twice the worst within-config spread (26.2 tok/s), so it is not explained by run-to-run variation at this sample size. The single-run 0.91× estimate holds up.

## Caveats

- **Repeats share a server process.** The model is loaded once per config and the benchmark run N times against the same live server. This measures benchmark-to-benchmark variance, **not** full cold-start variance — real deployment variance (load placement, memory layout, JIT state) could be larger.
- **Single concurrency (c=64).** The gap could differ at other batch sizes; this tests only the point the original comparison used.
- **Small n.** Three repeats bounds gross noise, not subtle systematic effects.

## Source data

| Per-repeat JSON / logs | `logs/atom/kimi_repeats_20261002_112609/<config>_rep<N>.{json,log}` |
|---|---|
| Server logs | `logs/atom/kimi_repeats_20261002_112609/<config>_server.log` |
| This table as CSV | `results/kimi-k3-repeats.csv` |
| Single-run comparison this tests | `kimi-k3-comparison.md` |

