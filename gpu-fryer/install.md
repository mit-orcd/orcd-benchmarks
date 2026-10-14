# gpu-fryer — installation

gpu-fryer is a Rust GPU stress test from Hugging Face. It runs here from a
prebuilt container image, so there is nothing to compile.

- Source: <https://github.com/huggingface/gpu-fryer> (a reference clone is in `gpu-fryer/`)
- Image: `docker://ghcr.io/huggingface/gpu-fryer:1.1.0`

```bash
cd /orcd/data/orcd/022/benchmarks/gpu-fryer
module load apptainer/1.4.2
singularity pull gpu-fryer_1.1.0.sif docker://ghcr.io/huggingface/gpu-fryer:1.1.0
mkdir -p output
```

The job scripts expect `gpu-fryer_1.1.0.sif` in this dir.

## NVML

The image does not include NVML (`libnvidia-ml.so.1`). The job scripts mount
the host `/lib64` into the container and point gpu-fryer at the library there:

```bash
singularity exec --nv -B /lib64:/home/$USER/lib64 gpu-fryer_1.1.0.sif \
    gpu-fryer --nvml-lib-path /home/$USER/lib64/libnvidia-ml.so.1 60
```

`job.sh`, `l40s.sh` and `test.sh` hard-code `/home/shaohao/lib64` as the mount
point. Other users should change it to `/home/$USER/lib64`.

## Check

Run a 60 s test on a GPU node, e.g. inside `salloc --gres=gpu:1`, with the
command above. The run should end with `All GPUs seem healthy`.
