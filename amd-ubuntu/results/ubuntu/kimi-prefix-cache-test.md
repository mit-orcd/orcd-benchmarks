# Prefix-cache test: vLLM recipe 2 on AMD's 128K/1K workload

Generated 2026-10-15 by `atom/analyze_prefix_test.py` from `logs/node6100/kimi-amd-recipe/atom/kimi_prefix_test_20261013_195317`, `logs/node6100/kimi-amd-recipe/atom/kimi_prefix_test_20261014_215900`, `logs/node6101/kimi-amd-recipe/atom/kimi_prefix_test_20261015_000338` (script `atom/run_kimi_prefix_test.sh`).

**Question.** With vLLM recipe 2 (`amd-kimi-k3-recipe.pdf`) our nodes match AMD's tested numbers from 8 users up but reach only 0.56x, 0.73x and 0.82x of them at 1, 2 and 4 users. The PDF's client sets no `--seed` (so every run sends the same prompts) and vLLM v0.29.0 keeps prefix caching on by default; 10–20 prompts of about 122K tokens fit in the 2.8M-token KV cache. If AMD's low-user runs followed an earlier run with the same prompts, most of the prompt processing would have come from the cache.

**Test.** Same image, server flags and client flags as our earlier run, one server: each user count is run twice with the same prompts (first run, then repeated run), and 1 user once more with new prompts. The prefix-cache hit rate is read from the server's counters before and after each test. 8–128 users were added later (2026-10-14, one server per node, new prompts per user count), to fill the table. TTFT = time to first token, TPOT = time per output token; metric = total tokens (input + output) per second per GPU, as in the PDF.

## Result

**Compared: the repeated run (same prompts again) vs the first run and vs AMD's tested numbers, same node, recipe and workload.** **Partly confirmed: the prefix cache explains part of the low-user gap.** Repeated prompts give 719 tok/s/GPU at 1 user (0.82x AMD's 874), faster than with new prompts but still below AMD's number, and 1,500 at 2 users (1.38x AMD's 1,090). AMD's 1- and 2-user numbers lie between our first and repeated runs, which fits runs with partly cached prompts. At 4 users the 40 prompts do not fit in the KV cache, so nothing is reused (also not in AMD's runs), and we stay at 0.82x AMD's number. At 8–128 users the repeated run gets 0% cache hits and 0.99–1.01x the first run, 1.01–1.10x AMD's numbers: the prompts do not fit in the KV cache, so the cache plays no role there. At 1 user the TPOT is 19.3 ms with or without the cache. So part of the gap has another cause (still to be profiled).

- **1 user:** repeated run 719 vs first run 489 tok/s/GPU (1.47x), 0.82x AMD's 874; prefix-cache hit rate 100%: the repeated run is faster, but still below AMD's number.
- **2 users:** repeated run 1,500 vs first run 781 tok/s/GPU (1.92x), 1.38x AMD's 1,090; prefix-cache hit rate 100%: the repeated run is faster than AMD's number.
- **4 users:** repeated run 1,019 vs first run 1,019 tok/s/GPU (1.00x), 0.82x AMD's 1,235; prefix-cache hit rate 0%: the repeated prompts were not found in the cache, so the cache cannot explain the gap here.
- **8 users:** repeated run 1,210 vs first run 1,195 tok/s/GPU (1.01x), 1.01x AMD's 1,199; prefix-cache hit rate 0%: the repeated prompts were not found in the cache, so the cache cannot explain the gap here.
- **16 users:** repeated run 1,211 vs first run 1,197 tok/s/GPU (1.01x), 1.03x AMD's 1,179; prefix-cache hit rate 0%: the repeated prompts were not found in the cache, so the cache cannot explain the gap here.
- **32 users:** repeated run 1,187 vs first run 1,190 tok/s/GPU (1.00x), 1.06x AMD's 1,118; prefix-cache hit rate 0%: the repeated prompts were not found in the cache, so the cache cannot explain the gap here.
- **64 users:** repeated run 1,202 vs first run 1,187 tok/s/GPU (1.01x), 1.10x AMD's 1,096; prefix-cache hit rate 0%: the repeated prompts were not found in the cache, so the cache cannot explain the gap here.
- **128 users:** repeated run 1,192 vs first run 1,205 tok/s/GPU (0.99x), 1.07x AMD's 1,110; prefix-cache hit rate 0%: the repeated prompts were not found in the cache, so the cache cannot explain the gap here.

| users | AMD tested | ours, 2026-10-07 sweep | ours, first run | ours, repeated run |
|---:|---:|---:|---:|---:|
| 1 | 874 | 487 | 489 | 719 |
| 2 | 1,090 | 791 | 781 | 1,500 |
| 4 | 1,235 | 1,015 | 1,019 | 1,019 |
| 8 | 1,199 | 1,212 | 1,195 | 1,210 |
| 16 | 1,179 | 1,189 | 1,197 | 1,211 |
| 32 | 1,118 | 1,227 | 1,190 | 1,187 |
| 64 | 1,096 | 1,191 | 1,187 | 1,202 |
| 128 | 1,110 | 1,188 | 1,205 | 1,192 |

## All tests

**Compared: each test vs AMD's tested number at the same users; ratio above 1 = ours faster.**

| test | users | prompts | seed | prompts seen before? | prefix-cache hit rate | TTFT median, ms | TPOT median, ms | total tok/s/GPU | vs AMD tested |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `c1_cold` | 1 | 10 | 0 | no (first run) | 0% | 9,869 | 19.3 | 489 | 0.56x |
| `c1_repeat` | 1 | 10 | 0 | **yes (same prompts again)** | 100% | 351 | 19.3 | 719 | 0.82x |
| `c1_seed1` | 1 | 10 | 1 | no (new seed) | 0% | 10,478 | 19.3 | 550 | 0.63x |
| `c2_cold` | 2 | 20 | 2 | no (first run) | 0% | 11,115 | 29.4 | 781 | 0.72x |
| `c2_repeat` | 2 | 20 | 2 | **yes (same prompts again)** | 100% | 391 | 20.4 | 1,500 | 1.38x |
| `c4_cold` | 4 | 40 | 3 | no (first run) | 0% | 11,156 | 51.4 | 1,019 | 0.82x |
| `c4_repeat` | 4 | 40 | 3 | **yes (same prompts again)** | 0% | 11,163 | 50.9 | 1,019 | 0.82x |
| `c8_cold` | 8 | 80 | 108 | no (first run) | 0% | 12,293 | 91.9 | 1,195 | 1.00x |
| `c8_repeat` | 8 | 80 | 108 | **yes (same prompts again)** | 0% | 12,547 | 91.9 | 1,210 | 1.01x |
| `c16_cold` | 16 | 160 | 116 | no (first run) | 0% | 46,595 | 163.9 | 1,197 | 1.02x |
| `c16_repeat` | 16 | 160 | 116 | **yes (same prompts again)** | 0% | 43,274 | 165.8 | 1,211 | 1.03x |
| `c32_cold` | 32 | 320 | 132 | no (first run) | 0% | 263,565 | 171.8 | 1,190 | 1.06x |
| `c32_repeat` | 32 | 320 | 132 | **yes (same prompts again)** | 0% | 265,609 | 171.9 | 1,187 | 1.06x |
| `c64_cold` | 64 | 640 | 164 | no (first run) | 0% | 689,307 | 174.2 | 1,187 | 1.08x |
| `c64_repeat` | 64 | 640 | 164 | **yes (same prompts again)** | 0% | 683,039 | 170.8 | 1,202 | 1.10x |
| `c128_cold` | 128 | 1280 | 228 | no (first run) | 0% | 1,532,206 | 170.1 | 1,205 | 1.09x |
| `c128_repeat` | 128 | 1280 | 228 | **yes (same prompts again)** | 0% | 1,543,173 | 172.8 | 1,192 | 1.07x |

- 4 users use 40 prompts (about 4.9M tokens), more than the KV cache holds, so few hits are expected there even when prompts repeat.
- A high hit rate with a short TTFT means the prompt was mostly read from the cache instead of being processed.

## What it means

- Prompts reused from the cache can make a 1–2 user number look much faster (1.5–1.9x here), so this part of the gap to `amd-kimi-k3-recipe.pdf` is a benchmarking effect, not a slower node. Fair comparisons should use new prompts for every run (a different `--seed`) or restart the server between runs.
- The cache does not explain everything: at 1 user (fully cached) and at 4 users (no reuse possible) we are still at about 0.82x AMD's number. This remaining gap is in the decode steps (TPOT) and is still to be profiled (e.g. rocprofv3).
- From 8 users up the prompts (80 or more of about 128K tokens) do not fit in the 2.8M-token KV cache, so a repeated run cannot reuse them and the cache does not change those numbers.
- It is still worth asking AMD whether their table was measured after an earlier run on the same server.
