# NCCL Benchmark Report on Server with 8 H200 GPUs

**Author:** Mikhail Serkov

## Overview

This report presents the results of NCCL benchmark tests conducted on a server equipped with 8 H200 GPUs. The tests were run on two different kernel configurations:

- **Generic Ubuntu Kernel**: `Linux 6.8.0-51-generic`
- **Nvidia Kernel**: `6.8.0-1019-nvidia`

Each benchmark test consists of 10 runs, and the results provided below show the average bandwidth (Algbw) in GB/s for each test across varying data sizes, now converted to human-readable formats.

## NCCL Tests

NCCL (NVIDIA Collective Communications Library) is a high-performance library for collective communication operations optimized for NVIDIA GPUs. It is used for multi-GPU and multi-node communication in high-performance computing environments.

Here are the tests run in this benchmark and a brief description of each:

1. **All_gather**: A communication pattern where all participants send data to all other participants.
2. **All_reduce**: A collective operation where each participant's data is combined and distributed back to all participants.
3. **All_to_all**: A pattern where each participant sends data to all other participants, and receives data from all participants.
4. **Broadcast**: A single participant sends data to all other participants.
5. **Gather**: A collection of data from all participants to a single participant.
6. **Hypercube**: A communication pattern optimized for multi-dimensional communication topologies.
7. **Reduce_scatter**: Combines data across participants and then scatters it among the participants.
8. **Reduce**: A common reduction operation where data is combined and reduced to a single value across all participants.
9. **Scatter**: A single participant sends portions of its data to all participants.
10. **Sendrecv**: A simple send and receive operation where each participant sends data to one other participant and receives data in return.

## Benchmark Results

### 1. All_gather

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 19.63                         | 20.35                       |
| 4MB     | 78.95                         | 77.75                       |
| 16MB    | 246.53                        | 245.6                       |
| 64MB    | 359.18                        | 360.15                      |
| 256MB   | 403.43                        | 403.68                      |
| 1GB     | 416.55                        | 416.0                       |
| 4GB     | 419.54                        | 419.44                      |
| 16GB    | 421.78                        | 421.73                      |

### 2. All_reduce

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 19.45                         | 19.08                       |
| 4MB     | 76.92                         | 78.55                       |
| 16MB    | 135.98                        | 135.99                      |
| 64MB    | 193.7                         | 193.31                      |
| 256MB   | 206.23                        | 203.53                      |
| 1GB     | 209.64                        | 209.44                      |
| 4GB     | 210.62                        | 210.6                       |
| 16GB    | 210.89                        | 210.91                      |

### 3. All_to_all

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 12.99                         | 13.68                       |
| 4MB     | 38.28                         | 40.0                        |
| 16MB    | 152.1                         | 154.95                      |
| 64MB    | 327.07                        | 320.36                      |
| 256MB   | 347.66                        | 359.61                      |
| 1GB     | 380.36                        | 380.45                      |
| 4GB     | 396.41                        | 397.29                      |
| 16GB    | 395.91                        | 397.66                      |

### 4. Broadcast

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 19.76                         | 18.45                       |
| 4MB     | 75.11                         | 23.5                        |
| 16MB    | 218.6                         | 217.58                      |
| 64MB    | 308.23                        | 306.91                      |
| 256MB   | 348.78                        | 349.02                      |
| 1GB     | 361.86                        | 359.81                      |
| 4GB     | 364.68                        | 364.89                      |
| 16GB    | 365.65                        | 365.2                       |

### 5. Gather

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 27.25                         | 25.77                       |
| 4MB     | 103.32                        | 92.49                       |
| 16MB    | 319.69                        | 317.08                      |
| 64MB    | 393.17                        | 391.79                      |
| 256MB   | 420.74                        | 420.49                      |
| 1GB     | 428.43                        | 428.36                      |
| 4GB     | 430.74                        | 430.73                      |
| 16GB    | 431.25                        | 431.26                      |

### 6. Hypercube

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 9.31                          | 5.17                        |
| 4MB     | 35.95                         | 16.66                       |
| 16MB    | 139.68                        | 144.19                      |
| 64MB    | 211.33                        | 212.88                      |
| 256MB   | 225.14                        | 224.5                       |
| 1GB     | 223.36                        | 227.19                      |
| 4GB     | 244.4                         | 237.09                      |
| 16GB    | 238.99                        | 239.6                       |

### 7. Reduce_scatter

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 19.75                         | 20.05                       |
| 4MB     | 77.39                         | 80.99                       |
| 16MB    | 231.66                        | 232.32                      |
| 64MB    | 346.0                         | 346.28                      |
| 256MB   | 381.18                        | 381.39                      |
| 1GB     | 399.65                        | 400.0                       |
| 4GB     | 412.29                        | 408.28                      |
| 16GB    | 416.08                        | 416.39                      |

### 8. Reduce

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 19.26                         | 19.07                       |
| 4MB     | 70.59                         | 70.66                       |
| 16MB    | 217.18                        | 219.19                      |
| 64MB    | 311.08                        | 311.12                      |
| 256MB   | 350.51                        | 351.22                      |
| 1GB     | 363.24                        | 363.83                      |
| 4GB     | 366.91                        | 366.82                      |
| 16GB    | 367.39                        | 367.29                      |

### 9. Scatter

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 25.23                         | 25.15                       |
| 4MB     | 90.62                         | 89.73                       |
| 16MB    | 313.99                        | 318.32                      |
| 64MB    | 384.83                        | 384.37                      |
| 256MB   | 413.74                        | 413.59                      |
| 1GB     | 420.51                        | 421.08                      |
| 4GB     | 423.96                        | 423.74                      |
| 16GB    | 424.23                        | 425.12                      |

### 10. Sendrecv

| Size    | Algbw (GB/s) - Generic Kernel | Algbw (GB/s) - Nvidia Kernel |
|---------|-------------------------------|-----------------------------|
| 1MB     | 27.0                          | 10.71                       |
| 4MB     | 104.06                        | 63.85                       |
| 16MB    | 131.89                        | 129.05                      |
| 64MB    | 141.35                        | 140.42                      |
| 256MB   | 358.43                        | 358.0                       |
| 1GB     | 363.91                        | 363.96                      |
| 4GB     | 366.13                        | 365.94                      |
| 16GB    | 366.35                        | 366.61                      |

## Summary

The benchmark results show that there is not much performance difference between the **Nvidia Kernel** and the **Generic Ubuntu Kernel**. While the Nvidia Kernel provides slight improvements in certain tests, such as **All_gather**, **Reduce_scatter**, and **Scatter**, the differences are minimal across most communication patterns. Overall, both kernels perform similarly, with no significant performance advantage for either one in the tested configurations.

## Benchmark Script

The following SLURM script was used to run the NCCL tests:

```bash
#!/bin/bash
#SBATCH -p mit_normal_gpu  # sched_system_all
#SBATCH -t 30
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:8
#SBATCH -J nvhpc-23.3-ompi3 
#SBATCH -o out.%x-%N-%J
#SBATCH -w node2433
#SBATCH --reservation=ubuntu_testing

job_name=$SLURM_JOB_NAME
BUILD_DIR=../build-$job_name

module purge
module use /orcd/home/001/mserkov/orcd/c7/scratch/u22.04/easybuild/modules/all
module load nvhpc/2023_233/nvhpc/23.3

mpirun hostname
which mpirun
which nvcc
echo "Bin dir = $BUILD_DIR"

MIN_SIZE=1M
MAX_SIZE=16G
FACTOR=4
GPUS_PER_TASK=8

echo "num_cpu = num_mpi_tasks = $SLURM_NTASKS"
echo "num_gpu_per_task = $GPUS_PER_TASK"

#export NCCL_DEBUG=INFO

for program in sendrecv_perf reduce_perf broadcast_perf gather_perf scatter_perf  reduce_scatter_perf all_gather_perf all_reduce_perf alltoall_perf hypercube_perf
do
   echo "%%%%%%%%% $program %%%%%%%%%%"
   mpirun -np 1 --mca btl_openib_warn_no_device_params_found 0 $BUILD_DIR/$program -b $MIN_SIZE -e $MAX_SIZE -f $FACTOR -g $GPUS_PER_TASK
done
