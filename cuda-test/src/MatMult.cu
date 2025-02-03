#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <cuda_runtime.h>

#define dim 10000
#define max_val 10
#define max_error 0.01
#define numThreads_x 16
#define numThreads_y 16

__global__ void mat_mult(float* dev_matrix, float* mat_1, float* mat_2) {
    /* Runs matrix multiplication
    * all matrices are flattened 2D matrices of size dim
    * should fill in values of dev_matrix
    */
    int i, j, k;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < dim && j < dim) {
        dev_matrix[i * dim + j] = 0.0;
        for(k = 0; k < dim; k++) {
            dev_matrix[i * dim + j] += mat_1[i * dim + k] * mat_2[k * dim + j];
        }
    }
    return;
}

void serial_mat_mult(float* ans_mat, float* mat_1, float* mat_2) {
    int i, j, k;
    for (i = 0;i < dim;i++) {
        for(j = 0;j < dim; j++) {
            ans_mat[i * dim + j] = 0.0;
            for(k = 0;k < dim; k++) {
                ans_mat[i * dim + j] += mat_1[i * dim + k] * mat_2[k * dim + j];
            }
        }
    }
    return;
}
// __host__ __device__ int matrixIndex(int r, int c) {
//     return r * dim + c;
// }

int main() {
    float *host_1, *host_2, *host_ans;
    float *dev_1, *dev_2, *dev_ans;
    struct timeval start, end, diff;

    host_1 = (float*)malloc(dim * dim * sizeof(float));
    host_2 = (float*)malloc(dim * dim * sizeof(float));
    host_ans = (float*)malloc(dim * dim * sizeof(float));
    cudaMalloc((void**)&dev_1, dim * dim * sizeof(float));
    cudaMalloc((void**)&dev_2, dim * dim * sizeof(float));
    cudaMalloc((void**)&dev_ans, dim * dim * sizeof(float));

    srand(time(NULL));
    int i, j;
    for(i = 0;i < dim; i ++) {
        for(j = 0; j < dim; j++) {
            host_1[i * dim + j] = ((float)rand() / RAND_MAX) * max_val;
            host_2[i * dim + j] = ((float)rand() / RAND_MAX) * max_val;
        }
    }
    cudaMemcpy(dev_1, host_1, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_2, host_2, dim * dim * sizeof(float), cudaMemcpyHostToDevice);
    
    int numBlocks_x = (dim + numThreads_x - 1) / numThreads_x;
    int numBlocks_y = (dim + numThreads_y - 1) / numThreads_y;
    dim3 gridSize(numBlocks_x, numBlocks_y);
    dim3 threadSize(numThreads_x, numThreads_y);
    
    gettimeofday(&start, NULL);
    mat_mult<<<gridSize, threadSize>>>(dev_ans, dev_1, dev_2);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);
    printf("Cuda code took %.8f seconds \n", diff.tv_sec + (double)diff.tv_usec / 1000000);

    cudaMemcpy(host_ans, dev_ans, dim * dim * sizeof(float), cudaMemcpyDeviceToHost);

    float* serial_ans = (float*)malloc(dim * dim * sizeof(float));
    gettimeofday(&start, NULL);
    serial_mat_mult(serial_ans, host_1, host_2);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);

    printf("Serial code took %.8f seconds \n", diff.tv_sec + (double)diff.tv_usec / 1000000);

    for(i = 0;i < dim; i++) {
        for(j = 0; j < dim; j++) {
            if(fabs(host_ans[i * dim + j] - serial_ans[i * dim + j]) > max_error) {
                printf("There was a calculation error \n");
                printf("host got: %f \n", host_ans[i * dim + j]);
                printf("serial got: %f \n", serial_ans[i * dim + j]);
                return 0;
            }
        }
    }
    printf("The calculation was correct \n");

    free(host_1); free(host_2); free(host_ans); free(serial_ans);
    cudaFree(dev_1); cudaFree(dev_2); cudaFree(dev_ans);
    
    return 0;
}
