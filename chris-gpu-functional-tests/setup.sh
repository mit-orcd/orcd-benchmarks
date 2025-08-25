#!/bin/bash


curl -L https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 -o micromamba
chmod +x micromamba

./micromamba create -y -p ./myenv python=3.13 uv -c conda-forge

eval "$(./micromamba shell hook --shell bash)"
micromamba activate ./myenv

uv pip install torch
uv pip install numpy
