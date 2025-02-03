#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define intervals 5000000000LL
#define numThreads 1024
#define USECPSEC 1000000

__global__ void solve(double *device_sums) {
    __shared__ double shared_sum[numThreads];
    long long int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= intervals) {
        return;
    }
    double inc = 1.0 / intervals;
    double x = (index + 0.5) * inc;
    shared_sum[threadIdx.x] = 4.0 * inc / (x * x + 1.0);
    __syncthreads();
    int i;
    for(i = numThreads / 2; i > 0; i >>=1) {
        if (threadIdx.x < i) { // numThreads needs to be a power of 2 to be accurate!
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        device_sums[blockIdx.x] = shared_sum[0];
    }
    return;
}

double solve() {
    long long int i;
    double inc = 1.0 / intervals;
    double sum = 0.0;
    for(i = 0;i < intervals; i++) {
        double x = (i + 0.5) * inc;
        sum += 4.0 * inc / (x * x + 1.0);
    }
    return sum;
}


int main() {
    double *host_sums;
    double *device_sums;
    long long int numBlocks = (intervals + numThreads - 1) / numThreads;

    host_sums = (double*)malloc(numBlocks * sizeof(double));
    cudaMalloc((void **)&device_sums, sizeof(double) * numBlocks);

    struct timeval start, end, diff;
    gettimeofday(&start, NULL);
    solve<<<numBlocks, numThreads>>>(device_sums);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    printf("Kernel execution time of this program is %.8f \n", diff.tv_sec + (double)diff.tv_usec / USECPSEC);
    cudaMemcpy(host_sums, device_sums, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(device_sums);
    double cuda_approx = 0.0; int i;
    for (i = 0; i < numBlocks; i++) {
        cuda_approx += host_sums[i];
    }
    free(host_sums);
    printf("Cuda approximation for pi with %lld rectangles: %.15f \n", intervals, cuda_approx);

    gettimeofday(&start, NULL);
    double serial_sum = solve();
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    printf("Serial execution time of this program is %.8f \n", diff.tv_sec + (double)diff.tv_usec / USECPSEC);
    printf("Serial approximation for pi with %lld rectangles: %.15f \n", intervals, serial_sum);
    return 0;
}
