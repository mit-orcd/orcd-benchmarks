# nvidia-hpc-benchmarks — installation

NVIDIA ships HPL, HPL-MxP, HPCG and STREAM-GPU as one prebuilt NGC container,
so there is nothing to compile. The input decks (`HPL-1GPU.dat` …
`HPL-8GPUs.dat`) are inside the image under
`/workspace/hpl-linux-x86_64/sample-dat/`.

- Container: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks>
- Image: `docker://nvcr.io/nvidia/hpc-benchmarks:25.04` (24.03 also kept)

```bash
cd /orcd/data/orcd/022/benchmarks/nvidia-hpc-benchmarks/run
module load apptainer/1.1.9
singularity pull hpc-benchmarks_25.04.sif docker://nvcr.io/nvidia/hpc-benchmarks:25.04
```

`run/run.sh` expects the image at `run/hpc-benchmarks_25.04.sif`. The root dir
holds another copy, plus `hpc-benchmarks_24.03.sif` for the older scripts.

The pull takes several GB of temp space. Point it at scratch, not `/tmp`:

```bash
export APPTAINER_TMPDIR=~/orcd/scratch/apptainer-tmp APPTAINER_CACHEDIR=~/orcd/scratch/apptainer-cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
```

## Check

```bash
srun -p mit_normal_gpu --gres=gpu:1 -n 2 -t 20 --mem=100G \
  singularity exec --nv hpc-benchmarks_25.04.sif \
  /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat
# look for the WC0 line and "PASSED"
```
