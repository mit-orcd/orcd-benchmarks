# Plan: use AMD Hyperloom to optimize Kimi-K3 inference on 8 × MI355X

Status: plan only. Written 2026-11-02 and revised the same day after user feedback. No work started.

Sources:
- Docs: https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html (v1.0.0: install, Slurm, how-to, optimization loop, kernel path, authentication, session output, multi-node, troubleshooting).
- Code: https://github.com/AMD-AGI/Hyperloom (README, `src/hyperloom/inference_optimizer/SKILL.md`).

Hyperloom 1.0.0 came out after Claude's knowledge cutoff. Everything below comes from reading those pages on 2026-11-02, so step 0 checks the main points again on the real install.

## 0. User decisions (2026-11-02)

- **ROCm:** ROCm 10.1 (`amd-software/rocm-10.1.0`, `ROCM_STACK=10.1`) for anything run outside a container. Inside a container, use the image's own ROCm. Never ROCm 7.2.4.
- **Containers:**
  - Run every "docker" job with apptainer through our docker shim (`common/bin/docker`).
  - Use Hyperloom's **docker mode**, so the shim stands in for docker.
- **Credentials:**
  - Use the user's own Claude Code login. No AMD gateway.
  - Claude asks the user to type a credential when one is needed, and never handles it in chat.
- **Internet:** every install step that needs the internet runs on the **login node**.
- **Token test first (2026-11-02):**
  - Run one small case first, to measure how many Claude tokens a session uses.
  - **No long runs until the user asks.** That means no Kimi-K3 optimization sessions (step 4); the 1 h Kimi baseline-only check and our own baseline fill (step 3) are fine.
- **Model (2026-11-02):** Opus 5.5 (`CLAUDE_MODEL=claude-opus-5-5`; also `GEAK_CLAUDE_MODEL` / `FORGE_CLAUDE_MODEL` if they are used).

## 1. Background and goal

Notes from the meeting with Ryan and Marko (AMD): Hyperloom is a platform that uses AI agents to run AMD's profiling tools, then optimizes both the framework and the low-level kernels. AMD said it raised tokens/s by about 38% in their case. You talk to it through a chat box, which is your agent account (Claude Code). The agent profiles the workload, edits the serving scripts and code, runs them and collects the results.

**What the docs and code show:**
- **How it runs:**
  - Hyperloom is a Claude Code skill plus a Python package (`pip install hyperloom-inference-optimizer==1.0.0 --target .`).
  - You open a workspace in Claude Code, run `/hyperloom-setup`, then paste a launch prompt that starts with `@hyperloom/inference_optimizer/SKILL.md` and gives the model, framework, GPU, TP, concurrency, ISL/OSL, goal and budget.
  - The agent then starts `python -m hyperloom.inference_optimizer.cli optimize ...` and checks on it every 300 s.
- **Phases:** PRELUDE (baseline, roofline/profile) → FRAMEWORK_AGENT → EXPLORE (server args and env variants, patches) → KERNEL_AGENT (GEAK by default, or KernelForge) → SWEEP (concurrency and ISL/OSL ladder) → CLOSE (report, `session_breakdown.json`). If budget is left after SWEEP, it can loop back.
- **Frameworks:** `sglang`, `vllm`, `atom`, `xdit`, `custom`. **ATOM is supported, single node only**, which is our fastest Kimi-K3 engine. Model classes include `moe_mla` (DeepSeek-style MoE with MLA attention, the same layout as Kimi-K3).
- **Tools it uses:**
  - **Magpie:** runs the baseline, profile, explore and sweep benchmarks.
  - **InferenceX:** the benchmark scripts and reference numbers, cloned at a pinned commit.
  - **Torch profiler traces:** `*.pt.trace.json.gz`, including ATOM's.
  - **TraceLens:** analyzes those traces and writes `tracelens/analysis.md`.
  - **Roofline:** `roofline_ceiling.py` / `roofline_snapshot.py`.
  - **`rocm-smi` / `amd-smi`:** GPU state.
  - **Other:** GEAK / KernelForge (kernel rewrites in HIP, Triton and FlyDSL), Quark (quantization) and IntelliKit (low-level GPU tools).
  - Setup applies a "ROCm profiler hotfix". The docs don't say how rocprofv3 or rocprof-compute are used; step 0 checks this.
- **Docker mode:**
  - Setup only writes `.env`. Hyperloom then starts one long-running container (`docker run -d --entrypoint tail ... -f /dev/null`) and runs everything inside it with `docker exec`.
  - The coordinator and Claude Code run on the host.
- **Safety:**
  - Each candidate change is reviewed by a Critic agent, then benchmarked and accuracy-gated. It is kept only if it wins, otherwise it is reverted.
  - After each KEEP, the full stack is measured again end to end.
- **How results are reported:**
  - `session_breakdown.json` has `final.throughput_tok_s_per_gpu`, `final.cumulative_gain_pct_validated` and `final.action_path` (the list of changes kept).
  - `optimizations.entries[].gain_pct` is the gain of each change against the session baseline; this is the only per-change figure that can be added up.
  - It also has `phase_timeline`, `capability_summary`, `kernel_lifecycle`, `telemetry` (paths to the profiles and traces) and `token_usage`.
- **Credentials (auth page):**
  - A Claude subscription token (`claude setup-token` → `CLAUDE_CODE_OAUTH_TOKEN`) is "a complete Anthropic side". With only the Anthropic side, Claude and GEAK run and Codex is off.
  - The page also says the Critic agent and some kernel tools use the OpenAI-side key. Step 0 checks whether `--critic-protocol anthropic` covers the Critic.
- **AMD's published gains (upstream pull requests):**
  - DeepSeek-V4.1-Flash on MI355X (vLLM): +53.9% total tok/s.
  - gpt-oss-120b on MI355X: +9.7%.
  - gemma-4-26B on MI355X: +9.0%.
  - Qwen3-32B-FP8 on MI300X: +314% against an untuned Triton default.
  - So "about 38%" depends on how tuned the starting point is.

**Goal:** run Hyperloom on our Kimi-K3 setup and check whether tokens/s improves over our current results. Explain which changes give the gain and why, using Hyperloom's profiling output (TraceLens, roofline, torch traces) plus our own rocprofv3 check.

## 2. Internet: install only, or also while running?

**Both.** Install needs GitHub, PyPI and the Claude Code download. A running session also needs the internet the whole time, because Hyperloom's agents call the Claude API at every step (planning, Critic reviews, patches, GEAK).

| Needs internet | When | Where it runs | Plan |
|---|---|---|---|
| Hyperloom wheel, Python deps (PyPI) | install | login node | `pip install --target` into `amd-software/hyperloom/` |
| InferenceX, Magpie, TraceLens, GEAK (GitHub) | install (the installer clones them if their paths are unset) | login node | clone at the pinned commits and set `INFERENCEX_PATH`, `MAGPIE_PATH`, `TRACELENS_ROOT`, and the GEAK path, so the installer doesn't clone on the node |
| Claude Code CLI | install | login node | install into `amd-software/claude-code/` (one self-contained binary) |
| Container images | install | none needed | already have the SIFs: `atom-dev-nightly_202610011450.sif` and `vllm-openai-rocm-v0.29.0.sif` |
| Model weights | none needed | – | `/scratch/Kimi-K3` on the nodes; check that the DSpark draft for run C is local too |
| **Claude API (`api.anthropic.com`)** | **whole session** | **node6100** | the node needs outbound HTTPS: **confirmed, see below** |
| InferenceX reference numbers (`--compare-against-gpu`) | session, optional | node6100 | leave off |
| Remote knowledge base (`KB_STORE_URL`) | session, optional | – | not used; local KB only |

Hyperloom runs the servers and benchmarks on the GPUs itself, so the coordinator and Claude Code must run on the GPU node, not the login node.

**Internet check (2026-11-02, `curl` over ssh, 10 s timeout, no proxy variables set):**

| URL | node6100 | node6101 |
|---|---|---|
| https://api.anthropic.com | 404 in 0.09 s (reached; 404 is normal for the bare root) | 404 in 0.11 s |
| https://github.com | 200 in 0.22 s | 200 in 0.31 s |
| https://pypi.org/simple/ | 200 in 0.57 s | 200 in 1.14 s |
| https://huggingface.co | 200 in 0.08 s | 200 in 0.11 s |
| https://claude.ai | 403 in 0.10 s (reached; blocks plain curl) | 403 in 0.11 s |

Both nodes have direct outbound HTTPS, so running sessions can reach the Claude API, and no proxy is needed. Install steps still run on the login node (user decision).

**Python version:** the node runs Ubuntu 24.04 (Python 3.12) and the login node runs RHEL 9. So on the login node, install for 3.12:
- `pip install --target ... --python-version 3.12 --only-binary=:all: --platform manylinux2014_x86_64`;
- or use a Python 3.12 venv on `/orcd/data`.

Step 1 checks the result with one `python3 -c "import hyperloom"` on node6100.

## 3. Current results to beat (ISL/OSL 1024/1024, TP8, one node)

From `results/kimi-k3-recipes-compared.md`:

| Users | ATOM (new image, no MTP) | ATOM + MTP (new image) | vLLM recipe 1 | vLLM recipe 2 (stock v0.29.0) |
|---:|---:|---:|---:|---:|
| 1, per-user tok/s | 64.0 | **256.8** | 65.1 | 55.1 |
| 8, throughput tok/s | 431 | **812** | 353 | 315 |
| 64, throughput tok/s | – | **2,463** | 1,344 | 1,228 |
| 128, throughput tok/s | – | – | **1,865** | 1,735 |
| 256, throughput tok/s | 2,602 (base image, `max-num-seqs` 256) | – | 2,522 | 1,747 |

- MTP = multi-token prediction (speculative decoding with the `Inferact/Kimi-K3-DSpark` draft).
- Per-user tok/s = 1000 / median TPOT (time per output token).
- The ATOM + MTP acceptance rate is not yet checked for correctness. See §7.

## 4. Which runs (in priority order)

| Run | Framework | Container (apptainer SIF) | Start config | Why |
|---|---|---|---|---|
| **A** | `atom`, no MTP | `atom-dev-nightly_202610011450.sif` | our `atom/run_atom_server.sh` settings | Cleanest test of framework + kernel tuning on our fastest engine |
| **B** | `vllm` | `vllm-openai-rocm-v0.29.0.sif` | vLLM recipe 2 (`atom/run_kimi_amdrecipe.sh`) | Stock and simple, the case AMD's published vLLM gains are most like |
| **C** | `atom` + MTP | `atom-dev-nightly_202610011450.sif` | `atom/run_kimi_atom_mtp.sh` | Only after the MTP output check in §7 passes |

### Can A, B and C be compared directly with the current results?

**Not directly, for three reasons:**
1. **Different client and workload.** Hyperloom measures with its own client (Magpie/InferenceX), and its headline is tok/s **per GPU** (`throughput_tok_s_per_gpu`). Ours is node-wide output tok/s from our own client (random prompts, range ratio 0.8, `--ignore-eos`, 10 × users prompts). Warm-up, prompt lengths and the number of prompts can differ.
2. **Missing points in our current results.**

   | Run | Current results we have (ISL/OSL 1024/1024) | Gap |
   |---|---|---|
   | A: ATOM, new image, no MTP | 1 user (64.0 per-user tok/s), 8 users (431 tok/s) | **64, 128, 256 users not run on this image** (256 only on the old image) |
   | B: vLLM recipe 2, v0.29.0 | 1, 8, 64, 128, 256 users | none; `max-num-seqs` 128 caps the 256-user point |
   | C: ATOM + MTP | 1, 8, 64 users | 128 and 256 not run. MTP output not checked. Draft length changes with load (6 at 1 user, 2 above), but Hyperloom tunes one config at 64 users |
3. **Results are about 3–4 weeks old.** Images and nodes are the same, but re-measuring on the same day removes drift.

**How the plan makes them comparable:**
- **Step 3 (new):** re-measure our own baseline for A, B and C at 1, 8, 64, 128 and 256 users with our scripts, on the same day as the sessions. This also fills the gaps in the table above.
- **Step 3:** check that Hyperloom's baseline matches ours within about 5% at 64 users. Its per-GPU number × 8 should match our node number, after checking whether it counts output or total tokens.
- **Step 5:** re-measure Hyperloom's final stack with our own client and scripts. Only this before/after pair, same client and same day, is reported as the gain.
- **Recipe 1 (re-tuned for each load) is left out of the comparison.** Each Hyperloom run ends with one fixed config, so it is compared with single-config setups.

**Settings for every run:**
- `--model /scratch/Kimi-K3 --gpu-type mi355x --tp 8 --model-class moe_mla --isl 1024 --osl 1024 --conc 64`.
- `--conc-sweep-concs 1,8,64,128,256`, the same points as our tables.
- `--precision`: the MXFP4 value, checked in step 0.
- `--max-hours 12` for run A, adjusted after we see how long A takes.
- `--target-gain 10`.

## 5. Steps

### Step 0: checks (read-only)
1. **Disk and memory:** read `/etc/motd.d`, `~/orcd/.quota` and the free space on node6100/node6101 (`/scratch`, `/dev/shm`).
2. **Internet from the nodes:** done on 2026-11-02; both nodes reach the Claude API, GitHub, PyPI and Hugging Face (§2).
3. **Read the code** (on the login node, after cloning):
   - `SKILL.md` (the rest of it), `assets/install.sh` and `assets/*atom*.yaml` / `assets/*vllm*.yaml`.
   - The auth code under `cli/credentials/`, to see what runs with an Anthropic-only setup (Critic, GEAK, kernel tools).
   - What the "ROCm profiler hotfix" changes, so no system files are touched. There is no sudo for installs.
   - Which profilers are run (rocprofv3, rocprof-compute) and with what flags.
   - How the accuracy gate is measured (dataset, threshold).
   - Every `docker` sub-command and flag Hyperloom uses, so the shim can be checked against them.
4. **Docker shim gaps.** `common/bin/docker` already supports `run -d`, `--name`, `-v`, `-e`, `-w`, `exec`, `logs`, `stop`, and persistent overlays (`CTR_OVERLAY`). It ignores `--device`, `--group-add` and `--shm-size`. It does **not** yet support:
   - `--entrypoint` (Hyperloom uses `--entrypoint tail ... -f /dev/null`);
   - maybe `docker ps`, `inspect` or `rm`, depending on what step 0.3 finds.

   Add the missing flags to the shim in a backward-compatible way, so existing scripts behave the same. Then test them with a dummy container.
5. **Where Hyperloom's code edits go:** Hyperloom edits the framework code inside the container. Give each run its own persistent overlay (`CTR_OVERLAY=hyperloom-A`, `-B`, `-C`), so edits survive container restarts and the shared SIFs are never changed. If ext3 overlays don't work on the node, the shim falls back to `--writable-tmpfs`, which loses edits on restart. If that happens, bind-mount a writable copy of the ATOM/vLLM source from `/orcd/data` instead.
6. **ROCm 10.1 on the host:**
   - Host-side tools (`amd-smi`, `rocm-smi`, our own rocprofv3) use `ROCM_STACK=10.1`.
   - Hyperloom's default SGLang images and `SGLANG_ROCM_EXTRA` are ROCm 7.2.4. Do not use them; we only use our own SIFs.
   - Record each image's ROCm version in the results.
7. **Skip host-level tools:** do not run `platform_audit_bmc.py` (it can create a BMC admin account), `partition_mode_sweep.py` (needs privilege), or any power/perf-level flags.

### Step 1: install (login node: everything that needs the internet)
1. **Claude Code CLI:** install into `amd-software/claude-code/`. node6100 has a node-local `$HOME`, so call it by absolute path.
2. **Hyperloom:** install `hyperloom-inference-optimizer==1.0.0` for Python 3.12 into `amd-software/hyperloom/`, which is the workspace. Clone InferenceX, Magpie, TraceLens and GEAK into `amd-software/hyperloom/deps/` at the commits Hyperloom pins.
3. **Credential (the user types it, not Claude):**
   - The user runs `! /orcd/data/.../claude-code/claude setup-token` on the login node. It opens a browser login and prints a long-lived token for the user's Claude subscription.
   - The user puts it in `amd-ubuntu/private/hyperloom/.env` as `CLAUDE_CODE_OAUTH_TOKEN=...` (chmod 600; `private/` is not synced). Claude never reads it back or prints it.
   - Leave `ANTHROPIC_API_KEY` unset; preflight warns if both are set. Set `HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV` as the auth code needs (step 0.3).
4. **Other `.env` settings** (written by `/hyperloom-setup`, or by hand):
   - `HYPERLOOM_RUN_MODE=docker`;
   - `USER_DATA_PATH=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/hyperloom`;
   - `CLAUDE_MODEL=claude-opus-5-5` (user decision; check the allowlist, `INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL`, if Hyperloom refuses it);
   - the dep paths from item 2.
5. **Caches and temp files:** `TMPDIR`, pip, Triton and AITER caches go under `~/orcd/scratch` or the node's `/scratch`, never `/tmp`.

### Step 2: small token test (node6100)
- **Purpose:** measure how many Claude tokens (Opus 5.5) a Hyperloom session uses before any long run.
- **Small case:** Qwen3-8B (or the smallest model the demo accepts), 1 GPU, `--tp 1 --conc 64 --isl 1024 --osl 1024 --max-hours 1 --no-kernel`. It still runs PRELUDE, the framework agent and the Critic, so the agent calls are real.
- **Record:**
  - `token_usage` from `session_breakdown.json` (input, output and cache tokens per agent and per phase);
  - wall time;
  - the subscription usage shown by `/usage` in Claude Code before and after.
- **Estimate, then stop and report:**
  - Scale the cost to a 12 h Kimi-K3 session: per-hour rate × hours.
  - Kimi-K3 runs fewer candidates per hour because of long restarts, so the token rate per hour may be lower.
  - Then stop and report the numbers. **Wait for the user** before any longer run.
- **Optional second small case (only if the user asks):** the same model with kernel work on (`--max-hours 1`), to see the extra cost of GEAK.

The original smoke test (the "3h" Qwen3-8B demo) is replaced by this 1 h test, since it checks the same chain.

**Before the run:**
- Check `import hyperloom` works with the node's Python 3.12, and that `claude` starts and logs in with the token.
- Hold the `/dev/shm/shaohao-gpu.lock` flock for the whole session. Hyperloom's gate IR-1 also needs all GPUs idle (< 1% VRAM).
- Qwen3-8B is the one small download, done on the login node into `amd-software/models/`.
- The test also checks the whole chain (shim, overlay, Claude API from the node, Critic, Magpie, TraceLens) before Kimi-K3.

### Step 3: check the baseline (node6100 and node6101; short runs, allowed before the go-ahead)
- **Our own baseline, same day (new):** run our scripts for A (ATOM new image, no MTP), B (vLLM recipe 2) and C (ATOM + MTP) at 1, 8, 64, 128 and 256 users. Do each twice, through the shim, about 2 h per arm. This fills the missing A points (64/128/256 on the new image) and C points (128/256).
- **MTP output check (before C):** greedy (temperature 0) output with MTP must equal the output without MTP on the same prompts.
- **Hyperloom's PRELUDE baseline:** run it for run A with `--no-kernel --no-framework-agent --max-hours 1` (baseline + profile only).
- **Compare the client:** Hyperloom uses Magpie/InferenceX as its client, so compare its baseline at 64 users with our `atom.benchmarks.benchmark_serving` number for the same settings.
- **Pass mark:** they must agree within about 5%. If they don't, find out why (prompt range ratio, warm-up, `--ignore-eos`, `max-num-seqs`) before trusting any gain.
- **Keep the profile outputs:** keep this baseline's TraceLens `analysis.md`, roofline and torch traces as the "before" profile.

### Step 4: optimization sessions (background)
- Launch run A, then B, then C, one at a time on node6100.
- Each runs as one background session (nohup/tmux), started by Claude Code with the Hyperloom skill. The agent polls once per check and does not block.
- Watch `state.json` and `session_breakdown.json`. Do not kill or restart the session's processes by hand (a Hyperloom rule). Resume with `--resume-from "$SESSION_DIR"` if it stops.
- **Long-running jobs:** these sessions are 12–24 h background processes. **Do not start any of them until the user asks** (user decision, 2026-11-02), and only after the token test in step 2 has been reported.

### Step 5: check the gain ourselves (node6101)
- **Rebuild the final stack:**
  - Take `final.action_path` and the final config (`runs/explore/final_config.with_envs.yaml`) plus any kept code edits (from the run's overlay, saved as a diff).
  - Rebuild the server with our own scripts (`run_atom_server.sh` / `run_kimi_amdrecipe.sh`) through the shim, with the edits applied.
- **Measure:** run our normal sweep (1, 8, 64, 128, 256 users, 10 × users prompts) with our own client. Report:
  - throughput tok/s;
  - per-user tok/s;
  - median TPOT (time per output token);
  - median TTFT (time to first token).
- **Repeats:** run the baseline and final configs 3 times each (past repeats were within about 2%).
- **Accuracy:** run GSM8K on the baseline and final servers. Also do a greedy-output check: the same prompts at temperature 0 must give the same text, unless a change is meant to change numerics (for example precision changes), in which case report the GSM8K change.

### Step 6: what made the gain (attribution)
1. **Hyperloom's own figures:** from `session_breakdown.json`, list each kept change with `gain_pct` and `gain_method`. Also give `validated_total_gain_pct` vs `ledger_total_gain_pct` and `unattributed_gain_pct`. Group the changes:
   - server flags (batching, `max-num-seqs`, CUDA graph sizes, chunked prefill);
   - environment variables (AITER switches, attention backend, all-reduce choice);
   - precision or quantization;
   - GEMM tuning (hipBLASLt / AITER tuned shapes);
   - kernel rewrites (GEAK: MoE, MLA attention, fused ops);
   - collective changes (all-reduce, all-gather).
2. **Our own one-at-a-time test:** add each kept change alone to the baseline, and also remove each change alone from the final stack. Measure each at 1 and 64 users on node6101.
3. **Before and after profiles:**
   - **TraceLens `analysis.md` (before vs after):** GPU time by op type (MoE GEMMs, dense GEMMs, MLA attention, all-reduce, norm/activation, idle gaps between kernels), and the top 10 kernels with their time and share.
   - **Roofline:** memory-bound vs compute-bound, and distance from the ceiling, for the main kernels before and after.
   - **Our own rocprofv3 check:** run `rocprofv3 --kernel-trace --stats` on one decode step, baseline vs final, as a second tool besides TraceLens.
     - Use the ROCm 10.1 rocprofv3, bound into the container. If it doesn't work with the image's runtime, use the image's own rocprofv3 and say so.
     - Start from `atom/analyze_profile.py` and `atom/analyze_trace_cpu.py`.
     - Use rocprof-compute only on the 1–2 kernels that changed most, if needed.
   - **Link each profile change to a kept change.** For example: "MoE GEMM time −X ms per step after the tuned-shapes change, which explains Y% of the gain."
4. **Compare with B200:** note any change that matches something NVIDIA already ships, since it changes the picture in `mi355x-vs-b200-kimi-k3.md`.

### Step 7: results file and push
- Write `results/hyperloom-kimi-k3.md` with these sections:
  1. Setup.
  2. Baseline check (Hyperloom client vs ours).
  3. Results (ours before, Hyperloom final, gain %) for runs A/B/C at 1, 8, 64, 128 and 256 users.
  4. Attribution table.
  5. Profiles before vs after.
  6. Accuracy.
  7. Cost: hours, and tokens from `token_usage`.
  8. Notes for admins and users.
- Plain wording: spell out TPOT and TTFT, and say "InferenceX", never "theirs".
- Add one line to `results/kimi-k3-recipes-compared.md` and `SUMMARY.md` pointing to it.
- Scripts go in `amd-ubuntu/hyperloom/`, logs in `logs/hyperloom/`, and credentials only in `private/hyperloom/` (not synced).
- Push with `sync-and-push.sh` from the login node. Kept code edits are published only as diffs, after a license check.

## 6. Time estimate

| Step | Time |
|---|---|
| 0 Checks + shim flags | ~0.5 day |
| 1 Install on login node + credential | ~2 h (needs the user for `claude setup-token`) |
| 2 Small token test (Qwen3-8B, 1 h) | ~2 h, then stop and report |
| 3 Baseline check + our own same-day baseline (A, B, C) + MTP check | ~1 day (Kimi-K3 load + AITER warm-up is slow) |
| 4 Sessions A, B, C | 12–24 h each, one after another on node6100; **only when the user asks** |
| 5 Our own re-measure | ~4 h per run on node6101 |
| 6–7 Attribution, profiles, md | ~1 day |

## 7. Risks and open points

- **Subscription limits:**
  - All Claude calls (orchestration, Critic, GEAK) bill against the user's Claude subscription.
  - A 12–24 h session may hit the plan's usage limits and pause.
  - Check `token_usage` after the smoke test, before planning the Kimi-K3 runs.
- **Anthropic-only setup:** Codex is off. If the Critic or a kernel tool needs the OpenAI side and `--critic-protocol anthropic` doesn't cover it, that part is skipped and the results say so. No AMD gateway.
- **MTP correctness:** ATOM + MTP's 4× gain at 1 user is not yet checked. Run the greedy MTP-vs-no-MTP check before run C.
- **Model size:** Kimi-K3 is about 1 T parameters. Every candidate needs a server restart (weight load plus AITER JIT, 30+ min cold start per AMD's notes), so a 12 h budget tests only a few candidates. Keep the AITER/Triton caches between rounds.
- **Fair comparison:** report both:
  - gain vs Hyperloom's own baseline;
  - gain vs our best number at each load (the bold numbers in §3).
- **Workload fit:** random prompts with `--ignore-eos` favour some changes (and MTP). Keep the same workload as our tables, and say so.
- **Changes made by the agent:** they stay in per-run overlays. Nothing touches the shared SIFs, `amd-software/rocm-*`, other users, or chris-* dirs. The shim change keeps old behaviour for existing scripts.

## 8. Questions and answers (2026-11-02)

1. **Usage limits:** run a small case first to measure token use (step 2). No long runs until the user asks.
2. **Model:** Opus 5.5 (`claude-opus-5-5`).
3. **Are A, B and C directly comparable with the current results?** Not directly. See "Can A, B and C be compared directly" in §4. The plan makes them comparable with a same-day re-measure using our own client (steps 3 and 5).
4. **Internet:** resolved. Both nodes reach the Claude API, GitHub, PyPI and Hugging Face directly (§2).

Still open: whether to run all of A, B and C, or start with A only (decide after the token test).

## 9. Done when

- We have a check of the gain at 1, 8, 64, 128 and 256 users, measured with our own client.
- Each kept change has a measured share of the gain and a profile-based reason.
- GSM8K and greedy-output checks pass.
- `results/hyperloom-kimi-k3.md` is pushed.
