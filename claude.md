# sync and push

original dir: /orcd/data/orcd/022/benchmarks
destination dir:  /orcd/data/orcd/022/shaohao/orcd-benchmarks

sync these dirs with two level down:
all-bench
megatron-lm
gpu-fryer
cuda
dataloader
fio
gpu-burn-r8
mpi-calc-pi
mpi-io
mpi-laplace
mpi-p2p
nccl-tests
numpy
nvidia-hpc-benchmarks
openmp
stream-amd
stream-intel

only sync these files:
run and job scripts
readme
all md files

Never sync these files:
container image
source code
binary files
output files
any files that is larger than 10M

Never modify or delete files in the original dir
overwrite and delete files in the destination dir and github remote

push to github remote from the destination dir

never modify or delete the directories starting with chris- in the github remote. 

