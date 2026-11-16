# Why step 4 gained so little, and how to fix it

Session A: `logs/hyperloom/Kimi-K3/20261110T012123Z-2c1aea52` (Kimi-K3, ATOM without MTP, 8 GPUs).
- **Result:** paused on Nov 11, 21:48 UTC, after 6 legs and 7.46 h of the 12 h budget.
- **Best gain:** **+0.33%** (1,666.6 vs 1,661.1 output tok/s/GPU at 64 users), from one setting, `--cudagraph-mode FULL`.
- **Checkpoint:** `logs/hyperloom/Kimi-K3/ckpt/20261110T012123Z-2c1aea52/20261111T215005Z-paused/`

The low gain is mostly not because there is nothing left to improve. Hyperloom's test runs could not start. All findings below come from the session's event log (`storage/coordinator.db`) and run folders.

## Why it gained so little

1. **Main cause: the server would not start with changed settings.**
   - **What happens (corrected Nov 12):** the reference-script parser keeps the quotes. They are lost in `remove_server_args` (`hyperloom/orchestrator/actions/executors/_grid_server_args.py`), which Hyperloom uses each time it removes a flag. It splits the flags with `shlex.split`, which strips the JSON's inner quotes, and its repair step cannot fix values with `*` or `[ ]`. (Earlier this note named `reference_script.py:274`. That was wrong.)
   - **Result:** `--online_quant_config '{"global_quant_config":"ptpc_fp8",...}'` reaches ATOM as `{global_quant_config:ptpc_fp8,...}`. That is not valid JSON, so ATOM exits with `invalid loads value`.
   - **Effect:**
     - Every server started with changed settings (the "explore" step) crashed in the first round.
     - The baseline server crashed at all 5 user levels in the final multi-user check, so every pass through the phases ended with `conc_sweep_failed`.
2. **Temp-folder clash.** Two variants (`moe_backend_mega`, `expert_parallel`) died with `OSError: [Errno 39] Directory not empty` on `amd-software/cache/tmp/atom-metrics-*`. That temp folder is on the shared network filesystem, where deleting folders can fail like this.
3. **GEAK, the kernel agent, never returned a result.**
   - **What happens:** its Claude agent replied "The workflow is running in the background; I'll wait for its completion notification" and then ended. Hyperloom could not read a result: `Could not parse a JSON workflow return`.
   - **Cost:** 2–3 h and most of the tokens in each leg, for no gain. It runs about 40 Claude helpers at once, which is why the 5-hour limits kept hitting.
4. **The framework agent's helpers could not read any framework code.** Both finished specialist tasks report the sandbox blocked reading `/app/ATOM`, aiter and the model's `config.json`. Claude's sandbox only allowed reads under the Hyperloom folder. 22 more specialist tasks were cancelled. So no code patches were proposed. Fix: allow these paths in the agents' sandbox (Claude Code `--add-dir`, or Hyperloom's own sandbox setting) and give them more than 600 s.
5. **The time budgets were used up.**
   - **First run:** the framework agent went over its time cap (8,696 s of 7,380 s). After the first 5-hour limit, about 8 h passed with no Claude.
   - **Each resume:** the framework and explore budgets were already spent, so Hyperloom skipped straight past the steps that test settings.
   - **Hyperloom's critic said it directly:** "The run is looping without benchmarking anything."

## How to fix it (for the rerun)

1. **Fix the quoting:** in `remove_server_args`, split with Hyperloom's JSON-safe splitter (done Nov 12, see [rerun-report.md](rerun-report.md)). Then do a quick check that one explore variant starts before running the whole session.
2. **Move ATOM's temp files to a local disk:** set `TMPDIR=/dev/shm/shaohao-tmp` (per node, cleaned at the end) instead of the shared filesystem.
3. **Turn GEAK off,** or give it a short time cap, until its result problem is fixed. This also saves most of the tokens.
4. **Start a fresh session (`RESUME=new`)** with a full 12 h. Session A's time budgets are spent and its history is all failed runs, so resuming it gives little. It stays saved and can still be resumed:

   ```bash
   ssh node6100 "USED_HOURS=7.46 nohup bash $H/hyperloom/run_step4_autoresume.sh >/dev/null 2>&1 </dev/null &"
   ```
5. **Use a quick check, then the full run:** first a 1-hour test to confirm the server starts, explore runs and the multi-user check passes. Then the 12-hour run with the auto-resume loop.

## What gain to expect

Our baseline is already tuned (FP8 KV cache, PTPC-FP8, 64 max sequences), so simple flag changes may give only a few percent.

Where the time goes, from GEAK's profile (torch profiler trace + TraceLens, 64 users, rank 0):

| Kernels | Share of GPU time |
|---|---|
| MoE expert GEMMs (`gemm1/gemm2_a16w4`: FP4 weights, BF16 activations) | about 47% |
| TP8 all-reduce (`aiter::cross_device_reduce_2stage`) | about 12% |
| Other GEMMs (`kernel_gemm_0`, FP4 GEMM, hipBLASLt) | about 20% |
| Attention, norm, quant, gating, top-k | the rest |

Candidates, each needing the GSM8K accuracy check against the baseline:
- **MoE activations in FP8 instead of BF16 (a8w4).** The expert weights are already 4-bit. Our quantization config leaves the experts out of PTPC-FP8 (`*block_sparse_moe.experts*`), so their activations stay BF16.
- **Expert parallel instead of TP8 for the MoE layers**, to cut the 12% all-reduce. Hyperloom tried this variant, but it died from the temp-folder clash, not from a real result.
- **MoE kernel tile tuning:** GEAK's alternatives gave +0.3% at 64 users and up to +6% on the large prefill batch (1,024 tokens).

## Other fix already made (Nov 11)

`guard.sh` stopped the run 3 times by mistake (legs 0, 3 and 5). Each new critic request repeats recent events, including the limit message from the previous leg, and the guard read that copy as a new limit. The fix:
- The guard now skips `critic-workdir/` and `robustness-workdir/`.
- It reads real limit errors from new `backend_error` events in `storage/coordinator.db`.

Replaying all 6 legs through the new check finds the 3 real limits and none of the 3 false ones. See [README.md](README.md).

## How to fix each problem (plan for the rerun)

### 1. The framework agent's helpers could not read code (sandbox)

- **What is known:** Hyperloom found the code (`discovered framework roots: atom=ok aiter=ok`). It is meant to give those folders to each Claude helper with `--add-dir`, in `bypassPermissions` mode (`orchestrator/specialists/subprocess_.py`). Yet both helpers said every read outside the Hyperloom folder was refused.
- **Not yet known:** the exact cause. The saved transcripts keep only the final summary, not the refused commands.
- **Fix:**
  1. Run one helper by itself for about 10 min, with full logging, to see the refused command and which directories it was given.
  2. Then fix the cause:
     - **Folders not passed:** set `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=/app/ATOM:<aiter path>:/scratch/Kimi-K3` before `optimize`.
     - **Claude's own settings blocking reads:** give the helpers a settings file that allows reading those paths.
  3. Raise the helper time limit above 600 s.
  4. Check that one helper can now read `/app/ATOM`, aiter and the model's `config.json`.

### 2. Make the profile complete and reliable

- Add a **rocprofv3** kernel trace next to the PyTorch trace. GEAK's parser already merges both (`parse_profile.py --rocprof-dir`), which gives true hardware times. The roofline report itself says "confidence low" today.
- Profile at **1, 64 and 256 users**, not only 64. The share of time in the MoE kernels and the all-reduce changes with the number of users.
- Do this in our own step 6 too. With TraceLens and the roofline, that gives a before/after picture for every change.
- Find out why GEAK's own baseline (2,049.55 output tok/s) differs from Hyperloom's (1,661.1 tok/s/GPU). The likely cause is a different benchmark client and metric (GEAK uses `random_range_ratio` 0.8 and 320 prompts), but this is not checked yet.

### 3. Test the candidates one at a time, with our own scripts and GSM8K

Each candidate: 3 repeats at 1, 8, 64, 128 and 256 users (1024 tokens in, 1024 out), plus a GSM8K check against the baseline score of 0.958.

| Candidate | How to test | Targets |
|---|---|---|
| MoE activations in FP8 (a8w4) | Remove `*block_sparse_moe.experts*` from the online quant config, or use ATOM's a8w4 MoE path if it has one | MoE GEMMs, about 47% of time |
| Expert parallel | Turn on ATOM's expert-parallel flag. It died before only because of the temp-folder clash, so fix `TMPDIR` first | all-reduce, about 12% |
| MoE tile tuning | Apply GEAK's best tile settings (+0.3% at 64 users, up to +6% on large prefill) | MoE GEMMs |

### Order for the rerun

1. Fix the quotes (`remove_server_args`), the temp folder (`TMPDIR` on `/dev/shm`) and the helpers' file access. The cause of the helpers' file access problem was found on Nov 12: `claude` was not on PATH in the container, so Hyperloom fell back to in-process helpers without `--add-dir`.

**Status:** started Nov 12. Results are in [rerun-report.md](rerun-report.md).
2. Run a 1-hour test to confirm the server starts, explore runs and the helpers can read code.
3. Run a fresh 12-hour session (`RESUME=new`) with GEAK capped or off.
4. Test the 3 candidates above by hand with our scripts.
