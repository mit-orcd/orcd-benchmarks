#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define dim 4096*4096
#define numThreads 1024

__global__ void solve(int *a, int *b, int *c, int s1, int s2) {
    int thd_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thd_index < dim) {
        c[thd_index] = s1 * a[thd_index] + s2 * b[thd_index];
    }
    return;
}

void solve_serial(int *a, int *b, int *c, int s1, int s2) {
    // vectors fo dimension dim
    int i;
    for (i = 0;i < dim;i++) {
        c[i] = s1 * a[i] + s2 * b[i];
    }
    return;
}

int* randArr(int d) {
    int* ans = (int*)malloc(d * sizeof(int));
    int i;
    for (i = 0;i < d;i++) {
        ans[i] = 1 + rand() % 100;
    }
    return ans;
}

int main() {
    srand(time(NULL));
    int* host_a = randArr(dim);
    int* host_b = randArr(dim);
    int* host_c = (int*)malloc(dim * sizeof(int));

    int s1 = 1;
    int s2 = 2;
    struct timeval start, end, diff;

    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, dim * sizeof(int));
    cudaMalloc((void **)&dev_b, dim * sizeof(int));
    cudaMalloc((void **)&dev_c, dim * sizeof(int));

    cudaMemcpy(dev_a, host_a, dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, dim * sizeof(int), cudaMemcpyHostToDevice);
    int numBlocks = (dim + numThreads - 1) / numThreads;

    gettimeofday(&start, NULL);
    solve<<<numBlocks, numThreads>>>(dev_a, dev_b, dev_c, s1, s2);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    printf("Cuda code took %.8f seconds \n", diff.tv_sec + (double)diff.tv_usec / 1000000);

    cudaMemcpy(host_c, dev_c, dim * sizeof(int), cudaMemcpyDeviceToHost);

    gettimeofday(&start, NULL);
    solve_serial(host_a, host_b, host_c, s1, s2);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    printf("Serial code took %.8f seconds \n", diff.tv_sec + (double)diff.tv_usec / 1000000);
    
    free(host_a); free(host_b); free(host_c);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    return 0;

}
