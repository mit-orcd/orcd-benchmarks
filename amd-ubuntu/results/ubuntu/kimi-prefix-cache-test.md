# Prefix-cache test: vLLM recipe 2 on AMD's 128K/1K workload

Generated 2026-10-14 by `atom/analyze_prefix_test.py` from `logs/node6100/kimi-amd-recipe/atom/kimi_prefix_test_20261013_195317` (script `atom/run_kimi_prefix_test.sh`, node6100).

**Question.** With vLLM recipe 2 (`amd-kimi-k3-recipe.pdf`) our nodes match AMD's tested numbers from 8 users up but reach only 0.56x, 0.73x and 0.82x of them at 1, 2 and 4 users. The PDF's client sets no `--seed` (so every run sends the same prompts) and vLLM v0.29.0 keeps prefix caching on by default; 10–20 prompts of about 122K tokens fit in the 2.8M-token KV cache. If AMD's low-user runs followed an earlier run with the same prompts, most of the prompt processing would have come from the cache.

**Test.** Same image, server flags and client flags as our earlier run, one server: each user count is run twice with the same prompts (first run, then repeated run), and 1 user once more with new prompts. The prefix-cache hit rate is read from the server's counters before and after each test. TTFT = time to first token, TPOT = time per output token; metric = total tokens (input + output) per second per GPU, as in the PDF.

## Result

**Compared: the repeated run (same prompts again) vs the first run and vs AMD's tested numbers, same node, recipe and workload.** **Partly confirmed: the prefix cache explains part of the low-user gap.** Repeated prompts give 719 tok/s/GPU at 1 user (0.82x AMD's 874), faster than with new prompts but still below AMD's number, and 1,500 at 2 users (1.38x AMD's 1,090). AMD's 1- and 2-user numbers lie between our first and repeated runs, which fits runs with partly cached prompts. At 4 users the 40 prompts do not fit in the KV cache, so nothing is reused (also not in AMD's runs), and we stay at 0.82x AMD's number. At 1 user the TPOT is 19.3 ms with or without the cache. So part of the gap has another cause (still to be profiled).

- **1 user:** repeated run 719 vs first run 489 tok/s/GPU (1.47x), 0.82x AMD's 874; prefix-cache hit rate 100%: the repeated run is faster, but still below AMD's number.
- **2 users:** repeated run 1,500 vs first run 781 tok/s/GPU (1.92x), 1.38x AMD's 1,090; prefix-cache hit rate 100%: the repeated run is faster than AMD's number.
- **4 users:** repeated run 1,019 vs first run 1,019 tok/s/GPU (1.00x), 0.82x AMD's 1,235; prefix-cache hit rate 0%: the repeated prompts were not found in the cache, so the cache cannot explain the gap here.

| users | AMD tested | ours, 2026-10-07 sweep | ours, first run | ours, repeated run |
|---:|---:|---:|---:|---:|
| 1 | 874 | 487 | 489 | 719 |
| 2 | 1,090 | 791 | 781 | 1,500 |
| 4 | 1,235 | 1,015 | 1,019 | 1,019 |

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

- 4 users use 40 prompts (about 4.9M tokens), more than the KV cache holds, so few hits are expected there even when prompts repeat.
- A high hit rate with a short TTFT means the prompt was mostly read from the cache instead of being processed.

## What it means

- Prompts reused from the cache can make a 1–2 user number look much faster (1.5–1.9x here), so this part of the gap to `amd-kimi-k3-recipe.pdf` is a benchmarking effect, not a slower node. Fair comparisons should use new prompts for every run (a different `--seed`) or restart the server between runs.
- The cache does not explain everything: at 1 user (fully cached) and at 4 users (no reuse possible) we are still at about 0.82x AMD's number. This remaining gap is in the decode steps (TPOT) and is still to be profiled (e.g. rocprofv3).
- It is still worth asking AMD whether their table was measured after an earlier run on the same server.
