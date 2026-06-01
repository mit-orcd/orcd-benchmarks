# tests on Lambda 1CC dual node 8-way H100 system

commands

```
ssh -i ~/.ssh/id_rsa  -l ubuntu  192.222.48.142

cd /home/ubuntu/cnh/nccl-bench/nccl-tests

ubuntu@alpha-test-16way-node-001:~/cnh/nccl-bench/nccl-tests$ git remote -v
origin	https://github.com/NVIDIA/nccl-tests.git (fetch)
origin	https://github.com/NVIDIA/nccl-tests.git (push)

export CUDA_HOME=/usr
export NCCL_HOME=/usr
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7a1

make -j 16 MPI=1

ubuntu@alpha-test-16way-node-001:~/cnh/nccl-bench/nccl-tests$ cat myhosts 
gpu01 slots=1
gpu02 slots=1

mpirun --hostfile myhosts -n 2 ./build/all_reduce_perf -b 8 -e 8192M -f 2 -g 8 -n 30

```

![image](https://github.com/user-attachments/assets/4099754a-4950-48e4-8e12-5f27ff727f2d)
