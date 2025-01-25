```

cnh@dgx-06:~/nccl-test/nccl-tests$ cat commands 
export CUDA_HOME=/home/mserkov/easybuild/software/CUDA/12.1.1/
export NCCL_HOME=/home/mserkov/easybuild/software/NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1/
export MPI_HOME=/home/mserkov/easybuild/software/OpenMPI/4.1.5-GCC-12.3.0/
export C_PATH=/home/mserkov/easybuild/software/GCCcore/12.3.0
export LD_LIBRARY_PATH=/home/mserkov/easybuild/software/CUDA/12.1.1/lib:/home/mserkov/easybuild/software/NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1/lib:/cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64:/usr/mpi/gcc/openmpi-4.1.7a1/lib

mpirun -x LD_LIBRARY_PATH --hostfile myhosts -n 2  ./build/all_reduce_perf -b 8 -e 16384M -f 2 -g 8 -n 30

cnh@dgx-06:~/nccl-test/nccl-tests$ cat myhosts 
dgx-06 slots=1
dgx-07 slots=1

cnh@dgx-06:~/nccl-test/nccl-tests$ pwd
/home/cnh/nccl-test/nccl-tests

```
