# Hyperloom step 2: small token test (2026-11-09)

This is step 2 of [plan-hyperloom.md](../plan-hyperloom.md). It runs one short Hyperloom session to measure how many Claude tokens a session uses and how long it takes.

**Setup:**
- Model and hardware: Qwen3-8B on 1 MI355X GPU on node6100.
- Software: the vLLM v0.29.0 image in apptainer.
- Workload: 64 users, 1024 tokens in and 1024 out, BF16.
- Hyperloom settings: 1-hour budget, no kernel agent, no framework agent.
- Claude model: Opus 5.5, using the user's own Claude Code login.
- Script: `hyperloom/run_step2.sh`.

## Try 2 (the full run)

| | |
|---|---|
| Run time | 46 min (it ended itself before the 1 hour budget) |
| Claude tokens | about 0.64M in total: 552K re-read from Anthropic's prompt cache (cheap), 76K written to that cache, 8.8K output |
| Claude calls | 7 (orchestrator 3, specialist 2, critic 1, robustness checker 1) |
| Starting speed (Hyperloom's own baseline) | 7,245 output tok/s on 1 GPU at 64 users; TTFT (time to first token) 467 ms; TPOT (time per output token) 8.4 ms |
| Variants tried | 2 of 4 planned; none kept |
| Gain | 0% |
| Why it stopped | `robustness_escalated`: Hyperloom's time-budget check ended the session because there was no gain with only about 16 min left |

## Try 1 (stopped early)

| | |
|---|---|
| Run time | 6 min |
| Why it stopped | Ray could not start because its socket path was too long. Fixed with `RAY_TMPDIR=/dev/shm/shaohao-ray` |
| Claude tokens | 2.54M in total (22 calls) |

## What this shows

- **The whole chain works on our nodes:** Claude login, critic, vLLM in apptainer, measurement, report and token accounting.
- **1 hour is too short to get any gain.** The baseline measurement alone took 16 min, and the explore phase ran over its own time share. Most of the hour is benchmark time, not Claude calls.
- **Token use is small.** A 12-hour Kimi-K3 session may use roughly 10–20M tokens. This is a guess, not a measurement.

## Raw outputs

These are under `logs/hyperloom/`, which is not pushed to GitHub.
- Run logs: `step2-run.log` (try 2) and `step2-run-try1.log` (try 1).
- Hyperloom report: `Qwen3-8B/20261109T181452Z-1a21a16f/reports/final.md`.
- Token counts: `Qwen3-8B/20261109T181452Z-1a21a16f/session_breakdown.json`.
