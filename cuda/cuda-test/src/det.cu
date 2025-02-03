#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define dim 64
#define numThreads 32

__device__ int index(int r, int c) {
    return r * dim + c;
}

__device__ void add_rows(float* device_m, int r1, int r2, float scale) {
    // adds the entries from r1 * scale into r2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        device_m[index(r1, idx)] += scale * device_m[index(r2, idx)];
    }
    return;
}

__global__ void add1(float* device_m) {
    //adds 1 to every entry
    float ratio = -1 * device_m[index(1, 0)] / device_m[index(0, 0)];
    add_rows(device_m, 0, 1, ratio);
    return;
}

int main() {
    float* host_m = (float*)malloc(dim * dim * sizeof(float));
    int i, j;
    for(i = 0; i < dim; i++) {
        for(j = 0; j < dim; j++) {
            host_m[i * dim + j] = (float)(i + j + 1);
        }
    }
    float* device_m;

    cudaMalloc((void**)&device_m, dim * dim * sizeof(float));
    cudaMemcpy(device_m, host_m, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = (dim + numThreads - 1) / numThreads;

    add1<<<blockSize, threadSize>>>(device_m);

    cudaMemcpy(host_m, device_m, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);

    printf("The entry that should be made 0 is equal to: %f \n", host_m[dim]);

    free(host_m); cudaFree(device_m);
    return 0;
}
