# Hyperloom install on the MI355X nodes

Hyperloom 1.0.0 is AMD's AI-agent inference optimizer (MIT license, AMD-AGI). This file shows how it was installed for [plan-hyperloom.md](../plan-hyperloom.md). How to run it is in [README.md](README.md).

**Where things are:**

| What | Path |
|---|---|
| Hyperloom package | `amd-software/hyperloom/` (`hyperloom-inference-optimizer==1.0.0`) |
| Settings and token | `amd-ubuntu/private/hyperloom/.env` (chmod 600, never pushed); `amd-software/hyperloom/.env` is a symlink to it |
| Container images | `amd-software/sif/atom-dev-nightly_202610011450.sif` (ATOM, ROCm 7.2.4) and `vllm-openai-rocm-v0.29.0.sif` (vLLM, ROCm 7.2.3) |
| Writable layers on top of the images | `amd-software/cache/overlay/hyperloom-atom.img` and `hyperloom-vllm.img` (40 GB each) |
| Models | `amd-software/models/Qwen3-8B` (step 2 test model), `/scratch/Kimi-K3` on node6100 and node6101 |
| Docker stand-in | `amd-ubuntu/common/bin/docker` (turns docker commands into apptainer) |
| Install logs | `amd-ubuntu/logs/hyperloom/install-{atom,vllm}.log`, `dl-qwen3-8b.log` |

`B=/orcd/data/orcd/022/benchmarks` below. Steps 1–4 run on the **login node**, which has internet. Step 5 runs on **node6100**.

## 1. Install the Hyperloom package

```bash
W=$B/amd-software/hyperloom; mkdir -p $W; cd $W
python3.12 -m pip install --no-deps hyperloom-inference-optimizer==1.0.0 --target .
```

`--no-deps` is used because Hyperloom's dependencies are installed inside each container in step 4, not on the login node.

## 2. Write the settings file

```bash
mkdir -p $B/amd-ubuntu/private/hyperloom; cd $B/amd-ubuntu/private/hyperloom
cat > .env <<'EOF'
CLAUDE_CODE_OAUTH_TOKEN="<PLEASE_FILL_IN>"
ANTHROPIC_BASE_URL=https://api.anthropic.com
CLAUDE_MODEL=claude-opus-5-5
USER_DATA_PATH=/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/hyperloom
HYPERLOOM_RUN_MODE=docker
HYPERLOOM_DOCKER_TARGET_HOST=node6100
HYPERLOOM_SKILL_PATH=/orcd/data/orcd/022/benchmarks/amd-software/hyperloom/hyperloom/inference_optimizer/SKILL.md
EOF
chmod 600 .env
ln -sf $PWD/.env $B/amd-software/hyperloom/.env
```

Get the token yourself:
1. Run `claude setup-token` in a terminal.
2. Paste the token inside the quotes in `.env`. The quotes are needed, because the file is read by bash.

This uses your own Claude Code login, not an AMD gateway. Never print the token or push this file. The `private/` folder is skipped by `sync-and-push.sh`.

## 3. Download the small test model (step 2 only)

```bash
T=~/orcd/scratch/hf-tools; python3.12 -m pip install -q --target $T huggingface_hub requests
PYTHONPATH=$T python3.12 -c 'from huggingface_hub import snapshot_download as s; s(repo_id="Qwen/Qwen3-8B", local_dir="'$B'/amd-software/models/Qwen3-8B", max_workers=4)' \
  > $B/amd-ubuntu/logs/hyperloom/dl-qwen3-8b.log 2>&1
```

Kimi-K3 is already in `/scratch/Kimi-K3` on both AMD nodes.

## 4. Install Hyperloom's runtime inside each container (login node)

`hyperloom/install_in_overlay.sh` runs Hyperloom's own `install.sh` inside a container image and saves the result in that image's writable layer. It creates the layer first if it is missing.

```bash
cd $B/amd-ubuntu/hyperloom; L=$B/amd-ubuntu/logs/hyperloom
./install_in_overlay.sh atom atom-dev-nightly_202610011450 --skip-kernel-agent > $L/install-atom.log 2>&1
./install_in_overlay.sh vllm vllm-openai-rocm-v0.29.0    --skip-kernel-agent > $L/install-vllm.log 2>&1
```

- **What it installs:** Magpie (benchmark harness, from GitHub), the InferenceX benchmark scripts, the Claude Agent SDK, and Python tools. The script ends by printing `install rc=0` and the SDK version.
- **Why `--skip-kernel-agent`:** that part starts a Ray head, so it must run on the GPU node. See step 5.
- **Other `install.sh` options:** `--check-only` and `--dry-run`.

## 5. Kernel-agent part (on node6100, done by the run scripts)

The kernel-agent part installs Ray, TraceLens and GEAK. It runs at the start of every session, inside the run scripts `run_step2.sh` and `run_step4.sh` (a full `install.sh` without `--skip-kernel-agent`). It is fast when everything is already installed.

Two settings are needed on our nodes:
- `RAY_TMPDIR=/dev/shm/shaohao-ray`: Ray's socket path must be shorter than 108 bytes. The first step 2 try stopped on this.
- `ulimit -Sn 65536`: a warning about this is harmless.

`run_step4.sh` also installs `flash-linear-attention` into the ATOM layer once, because Kimi-K3 needs it in that image.

## Known issues

- `apt-get install moreutils` fails inside the container, because there is no root. Hyperloom handles this itself: only its timestamped log lines are lost.
- `python3 -c "import magpie"` fails in both layers, even though Magpie is installed. Hyperloom finds Magpie through its own path, and step 2 ran its benchmarks fine.
- Hyperloom's accuracy check found no results in step 2 (`accuracy eval not found`). We check accuracy ourselves with GSM8K in step 5 of the plan.
- AMD nodes cannot push to GitHub. Push from the login node with `sync-and-push.sh`.
