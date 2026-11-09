# Hyperloom token usage (2026-11-09)

## Step 2 test

Test: Qwen3-8B, 1 MI355X GPU on node6100, vLLM v0.29.0 image, 64 users, 1024 in / 1024 out, 1 hour budget, no kernel agent.
Model: Claude Opus 5.5 (`claude-opus-5-5`) for the main agent and the reviewer.
Script: `hyperloom/run_step2.sh`. Numbers from Hyperloom's `session_breakdown.json` (`token_usage`).

## Tokens

| Run | Time | Model calls | Total tokens | Cache read | Cache write | Output |
|---|---|---|---|---|---|---|
| Try 1 (stopped: Ray did not start) | 6 min | 22 | 2.54M | 2.28M | 224K | 32K |
| Try 2 (full run) | 46 min | 7 | **0.64M** | 0.55M | 76K | 8.8K |

Try 2 by part: main agent 0.48M, specialist agents 0.14M, reviewer 14K, robustness checker under 1K.
New (uncached) input is tiny in both runs (56–168 tokens).

## What happened in try 2

- Hyperloom starting throughput: 7,245 tok/s per GPU.
- It tried 2 of its 4 planned setting changes; neither gave a gain. Each benchmark takes about 15 minutes, so the hour ran out.
- Stop reason `robustness_escalated`: Hyperloom shut the session down early when little time was left and no gain had been kept.
- The accuracy check wrote no results file; to fix before step 4.

## What it means

- Token use follows the number of model calls, not the time. Most of the hour was benchmark time.
- Rough guess for a 12-hour Kimi-K3 session: 8–30M tokens, 85–90% of them cheap cache reads.
- The kernel phase (GEAK), not used here, may add much more. Still unknown.
- Retries cost a lot: try 1's 6 minutes of Ray failures used 4 times the tokens of the whole of try 2.

Session files: `logs/hyperloom/Qwen3-8B/20261109T181452Z-1a21a16f/` (try 2), `logs/hyperloom/Qwen3-8B/20261109T180421Z-35163414/` (try 1).

## Which run to use for planning

- Normal case: use try 2. In a real Kimi-K3 session most of the time is benchmarking (server load 10–20 min, longer runs), so calls per hour are the same or fewer than try 2.
- Failure case: try 1. Fast failures plus retries make many calls in minutes, about 2–3M tokens per failure.

## Retry rule (all future runs)

If something fails, retry once (with a fix if one is clear). If it fails again, stop and report; do not keep retrying.
For Hyperloom: watch each new session early and stop it if it fails the same way twice.

## Expected tokens for step 4 (rough)

Session A (Kimi-K3, ATOM, 12 hours, kernel phase on):

| Part | Estimate | Based on |
|---|---|---|
| Setup and setting tests | 5–10M | try 2 pace, about 0.8M per hour |
| Failures (retry once, then stop) | 2–3M each, maybe 1–3 failures | try 1 |
| Kernel phase (GEAK) | 5–50M | not measured; coding agent with many calls |
| **Total for A** | **about 10–60M** | |

Sessions A, B and C (ATOM, vLLM, ATOM+MTP): about 30–150M in total.
About 85–90% of tokens are cheap cache reads; output is only about 1–2%.

To narrow it down: run A first and check the count after a few hours before starting B and C, or first run a 2-hour Qwen3-8B test with the kernel phase on (about 1–5M tokens).
