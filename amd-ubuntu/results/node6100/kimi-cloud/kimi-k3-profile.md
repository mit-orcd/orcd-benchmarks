# Kimi-K3 — profiler trace summary (next-step #2)

Replaces the residual-based estimate in `kimi-k3-improve.md` §3 with measured
kernel time. That section attributed ~80% of step time to "prefill + scheduling"
by subtracting estimated costs — this is the direct measurement instead.

Traces: `/orcd/data/orcd/022/benchmarks/amd-software/cache/traces/kimi_20261002_134733` (10 file(s), 339 MB) — kept off-repo,
driver log `kimi_profile_20261002_134733`.

## No GPU kernel events found

Parsed 544,277 trace events from `model_ts_20261002_135235_629.pt.trace.json` but none carried a GPU kernel category. The trace may be CPU-only, or the categories differ in this kineto version. Raw traces retained; try ATOM's own `tools/analyze_trace_summary.py`.

