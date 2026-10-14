# b200-kimi — installation

This benchmark serves Kimi-K3 with vLLM from one container image, using a model
that is already staged on the cluster. Nothing is compiled. All paths, nodes,
reservation and Slurm account are in `common/env.sh`. Edit that file first: the
reservation `RESV` and account `ACCT` in it are from 2026-08 and have expired.

| Piece | Location | How |
|---|---|---|
| vLLM image (CUDA 13, needs driver r580+) | `imag/vllm-openai_kimi-k3.sif` (~15 GB) | `./pull-image.sh` |
| Ray 2.57.0 (not in the image) | `pylibs/` | `pip --target`, `--no-deps` |
| Kimi-K3 weights (1.42 TiB, 96 shards, mxfp4) | `/orcd/compute/orcd/025/models/Kimi-K3` | already staged, read-only |
| Smoke model Qwen3-8B-FP8 (8.9 GB) | `models/Qwen3-8B-FP8` | `sbatch download-kimi.sh` |
| vLLM fixes | `patches/*.py` | bind-mounted at run time, nothing to install |

## 1. Container image (login node, once)

```bash
cd /orcd/data/orcd/022/benchmarks/b200-kimi
./pull-image.sh                 # = apptainer pull of docker://vllm/vllm-openai:kimi-k3, module apptainer/1.5.2
./pull-image.sh --verify        # recheck the sha256 against imag/vllm-openai_kimi-k3.sif.manifest
```

The script takes a lock so only one pull runs at a time, writes a manifest, and
deletes the OCI layer cache afterwards. No other script pulls the image; they
all fail if it is missing. `--force` refetches.

## 2. Ray (for the 2-node TP8 × PP2 runs)

The image lacks Ray, but already has all of Ray's dependencies. Install only
the Ray wheel next to it, using the image's own Python 3.12:

```bash
cd /orcd/data/orcd/022/benchmarks/b200-kimi
module load apptainer/1.5.2
apptainer exec --bind $PWD imag/vllm-openai_kimi-k3.sif \
    python3 -m pip download ray==2.57.0 --no-deps -d pylibs/wheels
apptainer exec --bind $PWD imag/vllm-openai_kimi-k3.sif \
    python3 -m pip install --no-deps --target pylibs pylibs/wheels/ray-2.57.0-cp312-cp312-manylinux2014_x86_64.whl
apptainer exec --bind $PWD --env PYTHONPATH=$PWD/pylibs imag/vllm-openai_kimi-k3.sif \
    python3 -c "import ray, vllm.v1.executor.ray_executor; print(ray.__version__)"
```

`common/env.sh` sets `PYLIBS` and calls Ray as `python3 -m ray.scripts.scripts`,
because the `pylibs/bin/ray` shebang is not portable.

## 3. Model weights

Kimi-K3 is **not** downloaded. The scripts read the staged copy in place, which
another user owns, through a `:ro` bind. To check it without reading the
weights:

```bash
source common/env.sh && check_model && echo OK     # 96 shards, 1,560,936,091,448 bytes
```

The smoke-tier model is used by the gate job:

```bash
sbatch download-kimi.sh smoke      # hf download Qwen/Qwen3-8B-FP8 -> models/Qwen3-8B-FP8 (uses the image's hf CLI)
```

## 4. Caches

`common/env.sh` puts `HF_HOME` (`.hf/`) and the other persistent caches under
this dir, never under `$HOME`. The Triton cache goes to a per-user `/tmp` path
inside the container. Nothing needs to be created by hand.
