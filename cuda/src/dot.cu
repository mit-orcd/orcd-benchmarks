//takes the dot product of two vectors a specified number of times
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define dim 1000000000
// when dim <= 100 million, the array is cached, leading to very fast cpu runtimes
// at this dimension, the cpu must access from lower memory levels, leading to poor performance
// the gpu has to read from global memory anyways, making this a fairer comparison
// when cpu can access cached memory speedup ~ 30x for gpu binary 
#define numThreads 32
#define USECPSEC 1000000


// __device__ void reduce(volatile double* blockSum, int tid) {
//     blockSum[tid] += blockSum[tid + 32];
//     blockSum[tid] += blockSum[tid + 16];
//     blockSum[tid] += blockSum[tid + 8];
//     blockSum[tid] += blockSum[tid + 4];
//     blockSum[tid] += blockSum[tid + 2];
//     blockSum[tid] += blockSum[tid + 1];
// }

// __global__ void solve_reduce_full_unroll(double* device_u, double* device_v, double* device_a) {
//     // Implementation of the final optimized version from Nvidia 
//     // didn't use the template for blockSize - perhaps can implement that optimization 
//     // assumed 1024 threads
//     __shared__ double blockSum[1024];
//     int thd_id = threadIdx.x + blockIdx.x * blockDim.x * 2;
//     if (thd_id < dim) {
//         blockSum[threadIdx.x] = device_u[thd_id] * device_v[thd_id];
//     }
//     if (thd_id + blockDim.x < dim) {
//         blockSum[threadIdx.x] += device_u[blockDim.x + thd_id] * device_v[blockDim.x + thd_id];
//     }
//     __syncthreads();
//     if (threadIdx.x < 512) {
//         blockSum[threadIdx.x] +=blockSum[threadIdx.x + 512]; 
//     }
//     __syncthreads();
//     if (threadIdx.x < 256) {
//         blockSum[threadIdx.x] +=blockSum[threadIdx.x + 256]; 
//     }
//     __syncthreads();
//     if (threadIdx.x < 128) {
//         blockSum[threadIdx.x] +=blockSum[threadIdx.x + 128]; 
//     }
//     __syncthreads();

//     if (threadIdx.x < 64) {
//         blockSum[threadIdx.x] +=blockSum[threadIdx.x + 64]; 
//     }
//     __syncthreads();

//     if (threadIdx.x < 32) {
//         reduce(blockSum, threadIdx.x);
//     }
//     if (threadIdx.x == 0) {
//         atomicAdd(device_a, blockSum[0]);
//     }
// }


__global__ void solve_binary_double_load(double* device_u, double* device_v, double* device_a) {
    // first reduces witin each block, then uses atomicAdd
    __shared__ double blockSum[numThreads];
    int thd_id = threadIdx.x + blockIdx.x * blockDim.x * 2;
    if (thd_id < dim) {
        blockSum[threadIdx.x] = device_u[thd_id] * device_v[thd_id];
    }
    if (thd_id + blockDim.x < dim) {
        blockSum[threadIdx.x] += device_u[blockDim.x + thd_id] * device_v[blockDim.x + thd_id];
    }
    __syncthreads();
    int offset;
    for(offset = numThreads / 2; offset >= 1; offset >>=1) {
        if (threadIdx.x < offset) {
            blockSum[threadIdx.x] += blockSum[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(device_a, blockSum[0]);
    }
    return;
}

__global__ void solve_binary(double* device_u, double* device_v, double* device_a) {
    // first reduces witin each block, then uses atomicAdd
    __shared__ double blockSum[numThreads];
    int thd_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thd_id >= dim) {
        return;
    }
    blockSum[threadIdx.x] = device_u[thd_id] * device_v[thd_id];
    __syncthreads();
    int offset;
    for(offset = numThreads / 2; offset >= 1; offset >>=1) {
        if (threadIdx.x < offset) {
            blockSum[threadIdx.x] += blockSum[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(device_a, blockSum[0]);
    }
    return;
}

__global__ void solve_atomic(double* device_u, double* device_v, double* device_a) {
    //uses atomicAdd to add every part of the dot product
    int thd_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thd_id < dim) {
        atomicAdd(device_a, device_u[thd_id] * device_v[thd_id]);
    }
    return;
}



double dot_serial(double* v1, double* v2) {
    int i;
    double ans = 0.0;
    for(i = 0; i < dim; i++ ) {
        ans += v1[i] * v2[i];
    }
    return ans;
}

int main() {

    double *host_v, *host_u;
    double *device_v, *device_u;
    double host_a, *device_a;
    struct timeval start, end, diff;
    host_u = (double*)malloc(dim * sizeof(double));
    host_v = (double*)malloc(dim * sizeof(double));
    if (host_u == NULL) {
        printf("Failed to allocate memory \n");
    }
    cudaMalloc((void**)&device_v, dim * sizeof(double));
    cudaMalloc((void**)&device_u, dim * sizeof(double));
    cudaMalloc((void**)&device_a, sizeof(double));
    host_a = 0.0f; 

    int i; srand((unsigned int) time(NULL));
    for(i = 0;i < dim;i++) {
        host_v[i] = (double)rand() / RAND_MAX;
        host_u[i] = (double)rand() / RAND_MAX;
    }

    gettimeofday(&start, NULL);
    double exp = dot_serial(host_v, host_u);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    printf("The dot product is %f \n", exp);
    printf("Serial code took %f seconds to run \n", diff.tv_sec + (double)diff.tv_usec / USECPSEC);


    int blocks = (dim + numThreads - 1) / numThreads;
    cudaMemcpy(device_v, host_v, dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_u, host_u, dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_a, &host_a, sizeof(double), cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);
    solve_atomic<<<blocks, numThreads>>>(device_u, device_v, device_a);
    cudaDeviceSynchronize();
    cudaMemcpy(&host_a, device_a, sizeof(double), cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);

    printf("Cuda atomicAdd got %f for the dot product \n", host_a);
    printf("Cuda atomicAdd took %f time to run \n", diff.tv_sec + (double)diff.tv_usec / USECPSEC);

    cudaMalloc((void**)&device_a, sizeof(double));

    gettimeofday(&start, NULL);
    solve_binary<<<blocks, numThreads>>>(device_u, device_v, device_a);
    cudaDeviceSynchronize();
    cudaMemcpy(&host_a, device_a, sizeof(double), cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);

    printf("Cuda atomicAdd with binary reduction got %f for the dot product \n", host_a);
    printf("Cuda atomicAdd with binary reduction took %f time to run \n", diff.tv_sec + (double)diff.tv_usec / USECPSEC);

    cudaMalloc((void**)&device_a, sizeof(double));
    blocks /= 2;

    gettimeofday(&start, NULL);
    solve_binary_double_load<<<blocks, numThreads>>>(device_u, device_v, device_a);
    cudaDeviceSynchronize();
    cudaMemcpy(&host_a, device_a, sizeof(double), cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);

    printf("Cuda atomicAdd with binary reduction, double load got %f for the dot product \n", host_a);
    printf("Cuda atomicAdd with binary reduction, double load took %f time to run \n", diff.tv_sec + (double)diff.tv_usec / USECPSEC);


    // cudaMalloc((void**)&device_a, sizeof(double));
    // host_a = 0.0;

    // gettimeofday(&start, NULL);
    // solve_reduce_full_unroll<<<blocks, numThreads>>>(device_u, device_v, device_a);
    // cudaDeviceSynchronize();
    // cudaMemcpy(&host_a, device_a, sizeof(double), cudaMemcpyDeviceToHost);
    // gettimeofday(&end, NULL);
    // timersub(&end, &start, &diff);

    // printf("Cuda atomicAdd with binary reduction with loop unrolling got %f for the dot product \n", host_a);
    // printf("Cuda atomicAdd with binary reduction with loop unrolling optimization took %f time to run \n", diff.tv_sec + (double)diff.tv_usec / USECPSEC);

    free(host_v); free(host_u); 
    cudaFree(device_v); cudaFree(device_u);

    return 0;
    

}