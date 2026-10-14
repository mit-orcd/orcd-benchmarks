# dataloader — installation

The benchmark is two Python scripts, `bw-llama.py` and `bw-imagenet.py`. It
needs a conda env with PyTorch and the datasets, copied onto each filesystem
you want to test.

## 1. Conda env `torch`

The job scripts run `module load miniforge/24.3.0-0; source activate torch`.
Tested with python 3.12.3, pytorch 2.4.0 (CUDA 12.4), torchvision 0.19.0,
pillow 10.3.0 and numpy 2.1.1.

```bash
module load miniforge/24.3.0-0
conda create -y -n torch -c pytorch -c nvidia -c conda-forge \
    python=3.12 pytorch=2.4.0 torchvision=0.19.0 pytorch-cuda=12.4 pillow numpy
source activate torch
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

The benchmark itself is CPU-only, so the CUDA build is not required.

## 2. LLaMA-style dataset (`wikipedia_tokenized.pt`)

This is a `torch.save`d list of 2048-token `int64` tensors (~380 MB), made with
`bak/download_tokenize_llama_dataset.py` (HF `datasets` + the
`hf-internal-testing/llama-tokenizer`), using only the Wikipedia source.

```bash
source activate torch
pip install datasets transformers tqdm
mkdir -p ~/orcd/scratch/dl-build && cd ~/orcd/scratch/dl-build
# keep only ("wikipedia", "20220301.en") in DATASET_NAMES and name the output wikipedia_tokenized.pt
sed -e '/("c4", "en"),/d' -e '/("arxiv_dataset", None),/d' -e '/("bigcode\/the-stack", None),/d' \
    -e 's/tokenized_llama_chunks.pt/wikipedia_tokenized.pt/' \
    /orcd/data/orcd/022/benchmarks/dataloader/bak/download_tokenize_llama_dataset.py > make_pt.py
python make_pt.py              # writes llama_dataset/wikipedia_tokenized.pt
```

Copy the file onto every storage target that `loop-llama.sh` tests:

```bash
for i in 002 004 005 006 007 008 009 010 011 012 013; do
  cp llama_dataset/wikipedia_tokenized.pt /orcd/scratch/orcd/$i/$USER/
done
```

`loop-llama.sh` hard-codes `/orcd/scratch/orcd/$i/shaohao/`. Change it to your
user name.

## 3. ImageNet (for `bw-imagenet.py`)

`bw-imagenet.py` reads `<root>/train/` as a torchvision `ImageFolder`. Copy the
ILSVRC tree from the shared dataset onto the filesystem under test:

```bash
mkdir -p /orcd/scratch/orcd/008/$USER/imagenet/images_complete
rsync -a /orcd/datasets/001/imagenet/images_complete/ilsvrc \
         /orcd/scratch/orcd/008/$USER/imagenet/images_complete/
```

Then set `dir=` in `job-imagenet.sh` to that path.

## 4. Output dirs

```bash
cd /orcd/data/orcd/022/benchmarks/dataloader
mkdir -p out-llama out-imagenet
```
