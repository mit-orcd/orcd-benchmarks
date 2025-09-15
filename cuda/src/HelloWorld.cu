#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sayHelloWorld(int *device_cnt) {
    printf("Hello World from Block %d and Thread %d \n", blockIdx.x, threadIdx.x);
    atomicAdd(device_cnt, 1);
    return;
}

#define N 16

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int host_cnt = 0;
    int *device_cnt;
    cudaMalloc((void **)&device_cnt, sizeof(int));
    cudaMemcpy(device_cnt, &host_cnt, sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    sayHelloWorld<<<4, 4>>>(device_cnt);
    cudaEventRecord(stop);
    cudaMemcpy(&host_cnt, device_cnt, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    printf("Number of cores: %d \n", host_cnt);

    cudaDeviceSynchronize();
    return 0;
}
