## Simple GPU functional tests

For checking system config, driver and kernel module presence. 


## To setup
```
curl -L https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 -o micromamba
chmod +x micromamba

./micromamba create -y -p ./myenv python=3.13 uv -c conda-forge

eval "$(./micromamba shell hook --shell bash)"
micromamba activate ./myenv

uv pip install torch
uv pip install numpy

```

## Run 

```
eval "$(./micromamba shell hook --shell bash)"
micromamba activate ./myenv

python torch/check_gpus.py 
```

```
> Number of GPUs available: 8
> GPU 0: NVIDIA H200 - Available for PyTorch
> GPU 1: NVIDIA H200 - Available for PyTorch
> GPU 2: NVIDIA H200 - Available for PyTorch
> GPU 3: NVIDIA H200 - Available for PyTorch
> GPU 4: NVIDIA H200 - Available for PyTorch
> GPU 5: NVIDIA H200 - Available for PyTorch
> GPU 6: NVIDIA H200 - Available for PyTorch
> GPU 7: NVIDIA H200 - Available for PyTorch
> CUDA is available. PyTorch can use the GPU(s).
```
