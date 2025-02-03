#include <cublas_v2.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define dim 10000
#define error 0.01
#define USECPSEC 1000000

void serial_mat_mult(double* mat_1, double* mat_2, double* ans_mat) {
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

int main() {
    //single precision matrix multiplication first
    // will benchmark later
    double *host_a, *host_b, *host_c;
    double *dev_a, *dev_b, *dev_c;

    host_a = (double*)malloc(dim * dim * sizeof(double));
    host_b = (double*)malloc(dim * dim * sizeof(double));
    host_c = (double*)malloc(dim * dim * sizeof(double));
    cudaMalloc((void **)&dev_a, dim * dim * sizeof(double));
    cudaMalloc((void **)&dev_b, dim * dim * sizeof(double));
    cudaMalloc((void **)&dev_c, dim * dim * sizeof(double));

    srand((unsigned int) time(NULL)); int i; int j;
    for(i = 0; i < dim * dim; i++) {
        host_a[i] = (double)rand()/RAND_MAX;
        host_b[i] = (double)rand()/RAND_MAX;
    }

    cudaMemcpy(dev_a, host_a, dim * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, dim * dim * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle; cublasCreate(&handle);
    double scalar_1 = 1.0f;
    double scalar_2 = 0.0f;
    struct timeval start, end, diff;

    cudaEvent_t start_event, stop_event;
    float elapsed_time;

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, 0);
    // cuBLAS stores arrays in column major, not row-major, order
    cublasDgemm(handle, 
                CUBLAS_OP_T, CUBLAS_OP_T,
                dim, dim, dim,
                &scalar_1,
                dev_a, dim,
                dev_b, dim,
                &scalar_2, 
                dev_c, dim
    );
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    printf("cuBLAS took %f milliseconds \n", elapsed_time);

    cudaMemcpy(host_c, dev_c, dim * dim * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);



    double* serial_ans = (double*)malloc(dim * dim * sizeof(double));
    gettimeofday(&start, NULL);
    serial_mat_mult(host_a, host_b, serial_ans);
    gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);

    printf("serial code took %f seconds \n", diff.tv_sec + (double)diff.tv_usec / USECPSEC);

    for(i = 0;i < dim; i++) {
        for(j = 0; j < dim; j++) {
            // the result is transposed
            if(fabs(host_c[j * dim + i] - serial_ans[i * dim + j]) > error) {
                printf("There was a calculation error \n");
                printf("The error percent was %f \n", (fabs(host_c[i * dim + j] - serial_ans[i * dim + j]))/serial_ans[i * dim + j]);
                return 0;
            }
        }
    }
    printf("The calculation was correct \n");
    free(host_a); free(host_b); free(host_c); free(serial_ans);
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

    return 0;
}
