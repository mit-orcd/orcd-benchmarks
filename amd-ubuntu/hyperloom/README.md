# Running Hyperloom on Kimi-K3 (MI355X)

These are the commands for each step of [plan-hyperloom.md](../plan-hyperloom.md). How to install everything is in [install.md](install.md).

`B=/orcd/data/orcd/022/benchmarks`, `H=$B/amd-ubuntu`.

**Rules for every step:**
- Run each step on an AMD node (node6100 or node6101) with `nohup`. The run keeps going if you log out.
- Each script takes the GPU lock `/dev/shm/shaohao-gpu.lock`, so only one GPU job runs on a node at a time. If the lock is busy, the script exits.
- Each script writes a short progress file `logs/hyperloom/<step>-<node>.state` and a full log in `logs/hyperloom/`.
- Push results from the **login node**, because AMD nodes cannot push:

```bash
$B/sync-and-push.sh "<message>"
```

## Files in this folder

| File | What it does |
|---|---|
| `install_in_overlay.sh` | Installs Hyperloom's runtime inside a container image (see [install.md](install.md)) |
| `run_baselines.sh` | Step 3: our own baselines, ATOM with and without MTP, and vLLM (2 repeats) |
| `greedy_capture.py` | Step 3: saves the exact text of fixed prompts at temperature 0 (the MTP check) |
| `run_step2.sh` | Step 2: small token test (Qwen3-8B, 1 GPU, 1 hour) |
| `kimi_atom_reference.sh` | Step 4: our ATOM server settings, which Hyperloom uses as its starting setup |
| `run_step4.sh` | Step 4 session A: Kimi-K3, ATOM without MTP, 8 GPUs, 12 hours (`RESUME=<session dir>` to resume) |
| `guard.sh` | Started by `run_step4.sh`: checkpoint every 30 min; stops everything at a Claude limit, no auto-resume |
| `run_step4_autoresume.sh` | Same as `run_step4.sh`, but checks every 60 min after a Claude limit and resumes by itself when the limit is over |

## Step 2: small token test

```bash
ssh node6100 "nohup bash $H/hyperloom/run_step2.sh >/dev/null 2>&1 </dev/null &"
cat $H/logs/hyperloom/step2-node6100.state         # progress
tail -30 $H/logs/hyperloom/step2-run.log           # full log
```

- **Run:** Qwen3-8B on 1 GPU, 64 users, 1024 tokens in and 1024 out, 1 hour, with the kernel agent and framework agent turned off. Set `HOURS=2` to change the budget.
- **Output:** a session folder `logs/hyperloom/Qwen3-8B/<time>-<id>/` with:
  - `reports/final.md`: the report
  - `session_breakdown.json`: Claude token use under `attribution` and `by_component`
- **Results:** [results/hyperloom-step2.md](../results/hyperloom-step2.md).

## Step 3: our own baselines

```bash
ssh node6101 "nohup bash $H/hyperloom/run_baselines.sh >/dev/null 2>&1 </dev/null &"
cat $H/logs/hyperloom/baselines-node6101.state
```

The script runs these, twice each, at 1, 8, 64, 128 and 256 users, 1024 tokens in and 1024 out:
- ATOM without MTP
- ATOM with MTP: 6 draft tokens at 1 user, 2 above
- vLLM with AMD's recipe

**Output:** `logs/<node>/hyperloom-baseline/atom/kimi_atom_mtp_*` (ATOM) and `kimi_amdrecipe_isl1k_*` (vLLM).
- One JSON file per concurrency, named `*_c<users>.json`.
- `greedy.json` holds the MTP text check. This check did not work, because ATOM gives different text for the same prompt even at temperature 0. Step 5 uses GSM8K accuracy instead.

## Step 4: Hyperloom optimization sessions

**Session A** (Kimi-K3, ATOM without MTP, 8 GPUs, 12 hours):

```bash
ssh node6100 "nohup bash $H/hyperloom/run_step4.sh >/dev/null 2>&1 </dev/null &"
cat $H/logs/hyperloom/step4-node6100.state
tail -30 $H/logs/hyperloom/step4-A-run.log
```

The script does three things:
1. Installs `flash-linear-attention` once, and runs the full Hyperloom install with the kernel-agent part.
2. Runs:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
  --framework atom --model /scratch/Kimi-K3 --gpu-type mi355x --tp 8 --model-class moe_mla \
  --conc 64 --isl 1024 --osl 1024 --precision bf16 --max-model-len 16384 \
  --reference-script kimi_atom_reference.sh \
  --enable-conc-sweep --conc-sweep-concs 1,8,64,128,256 \
  --target-gain 10 --max-hours 12
```

3. Stops Ray at the end.

**How Hyperloom gets our server settings:** `--reference-script` gives Hyperloom our step 3 ATOM flags as its starting setup. These include FP8 KV cache, 64 max sequences and PTPC-FP8 online quantization. The JSON in that file has no spaces, because Hyperloom re-joins the flags with spaces and drops the quotes.

**Check in the first hour:** Hyperloom's starting number at 64 users should be within about 5% of our step 3 number, about 1,605–1,652 output tok/s. If it is not, stop the session (below) and find out why.

```bash
grep -m1 'baseline' $H/logs/hyperloom/Kimi-K3/*/reports/final.md
```

**Output:** `logs/hyperloom/Kimi-K3/<time>-<id>/`:
- `reports/final.md`: summary, best setup and gain
- `current_setting.sh`: the best server settings found
- `session_breakdown.json`: gain for each change and token use
- `patches/`: code patches, if any

**Stop a session early:**

```bash
ssh node6100 'pkill -f hyperloom.inference_optimizer.cli; sleep 30; pkill -f atom.entrypoints; ray stop --force'
```

**Checkpoints and Claude limits (`guard.sh`):** `run_step4.sh` starts `guard.sh` next to the session.
- **Every 30 min**, the guard copies the files needed to resume into `logs/hyperloom/Kimi-K3/ckpt/<session>/<UTC time>/`:
  - `state.json`
  - `storage/`, the event log that Hyperloom replays on resume
  - `reports/`, `current_setting.sh`, `session_breakdown.json` and `manifest.json`

  It keeps the newest 6 copies, about 2 MB each. It takes one more copy when the session ends.
- **Every 60 s**, it looks for a Claude limit message (usage limit, 5-hour or weekly limit, HTTP 429) in the run log and the session's LLM call logs. On a hit it:
  1. takes a checkpoint
  2. stops Hyperloom, the ATOM server and Ray
  3. writes `STOPPED_TOKEN_LIMIT` into the session folder
  4. writes a line to `step4-node6100.state` (the run ends with `guard rc=3`)

  **It never resumes by itself.**

**Resume a stopped session** (only when you decide to):

```bash
S=$H/logs/hyperloom/Kimi-K3/<session dir>          # the folder with state.json
ssh node6100 "RESUME=$S nohup bash $H/hyperloom/run_step4.sh >/dev/null 2>&1 </dev/null &"
```

The resume runs `optimize ... --resume-from $S`, which replays `state.json` and the event log, and writes a new log `step4-A-run-resume-<time>.log`. If `state.json` in the session folder is damaged, copy `state.json` and `storage/` back from the newest `ckpt/` copy first.

**Two ways to run step 4.** Choose one each time:

| Script | At a Claude limit |
|---|---|
| `run_step4.sh` | Checkpoint and stop. It resumes only when you start it again with `RESUME=`. |
| `run_step4_autoresume.sh` | Checkpoint and stop, check every 60 min whether the limit is over, then resume by itself. Repeats until step 4 is completed. |

```bash
ssh node6100 "nohup bash $H/hyperloom/run_step4_autoresume.sh >/dev/null 2>&1 </dev/null &"
cat $H/logs/hyperloom/step4-autoresume-node6100.state      # what the loop is doing
touch $H/logs/hyperloom/AUTORESUME_STOP                    # cancel the loop (checked every minute)
```

How the auto-resume loop works:
- **It continues the last run.** By default it resumes the newest session, `Kimi-K3/20261110T012123Z-2c1aea52` (session A). That session's baseline, best setting and history are kept. Use `RESUME=new` to start a new session instead.
- **Before every run, it checks that Claude can be used.** It sends one tiny Claude request: the same token and model as Hyperloom, no tools, a one-word reply. If the limit is still active, it checks again every 60 min. It holds no GPUs while waiting.
- **Each run** is `run_step4.sh` with `--resume-from`, so the checkpoints and the stop at a limit are the same as above.
- **After a stop at a limit,** it writes `stop_reason=token_limit` into `state.json`, keeping a backup copy. Without this, Hyperloom would count the waiting hours against its budget, and the next run would stop at once. The next run gets the hours that are left: 12 h in total. Session A used about 1.75 h with Claude before its limit, so the first run gets about 10.25 h.
- **It repeats** (limit → hourly check → resume) **until step 4 is completed**, which means Hyperloom ends by itself without a limit stop.
- **Safety limits:** 20 resumes (`MAX_RESUMES`), 7 days (`DEADLINE_HOURS`), or `touch AUTORESUME_STOP` to cancel.
- **Other settings:** `TOTAL_HOURS` (12), `USED_HOURS` (1.75), `CHECK_MIN` (60).

**Why `guard.sh` reads the whole session folder:** in session A, Hyperloom caught the limit error itself and kept running about 8 hours without Claude. The error message was only in `critic-workdir/*/request.json` and in GEAK's trace files. The guard now reads new text in every file of the session folder each minute. It skips `runtime/`, and it reads only text written since its last scan, so old limit messages from session A do not stop a resumed run.

**Fix on Nov 11:** the guard also skips `critic-workdir/` and `robustness-workdir/`. Each new critic request repeats recent event history, so the limit message from the previous run showed up again as new text. This caused 3 false stops: legs 0, 3 and 5, minutes after the hourly check had said Claude works. The guard now also reads new `backend_error` events in `storage/coordinator.db`, which is Hyperloom's own log. Replaying the 6 legs through the new check finds the 3 real limits and none of the false ones.

**Sessions B (vLLM) and C (ATOM with MTP)** run after A, if A goes well. Use the same script with `--framework vllm` and the vLLM image (B), or with the MTP flags added to the reference script (C).

## Step 5: measure the best setup again with our own scripts

Take the best settings from step 4 (`current_setting.sh`) and run them with our own client, `atom/run_kimi_atom_mtp.sh`:
- 3 repeats
- 1, 8, 64, 128 and 256 users
- a GSM8K accuracy check against the baseline

Our numbers, not Hyperloom's own, are the ones we report.

## Step 6: find which changes made the gain

1. Read the gain list in `session_breakdown.json`.
2. Add or remove one change at a time and measure each with our scripts.
3. Compare profiles before and after: TraceLens, roofline and rocprofv3.

## Step 7: write up and push

Write `results/hyperloom-kimi-k3.md`, then push from the login node with `sync-and-push.sh`.
