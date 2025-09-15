// omp_device_to_device_bandwidth.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define SIZE (1 << 28)  // 64 MB
#define REPEAT 100

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    int dev0 = 0, dev1 = 1;
    double *src, *dst;

    // Allocate memory on device 0 and device 1
    src = (double *) omp_target_alloc(SIZE * sizeof(double), dev0);
    dst = (double *) omp_target_alloc(SIZE * sizeof(double), dev1);

    if (!src || !dst) {
        printf("Failed to allocate device memory.\n");
        return 1;
    }

    // Initialize source data on device 0
    #pragma omp target teams distribute parallel for device(dev0) is_device_ptr(src)
    for (size_t i = 0; i < SIZE; ++i) {
        src[i] = (double)i;
    }

    // Warm-up copy
    omp_target_memcpy(dst, src, SIZE * sizeof(double), 0, 0, dev1, dev0);

    // Timed copy
    double start = get_time_sec();
    for (int i = 0; i < REPEAT; ++i) {
        omp_target_memcpy(dst, src, SIZE * sizeof(double), 0, 0, dev1, dev0);
    }
    double end = get_time_sec();

    double total_bytes = (double)SIZE * sizeof(double) * REPEAT;
    double bandwidth = total_bytes / (end - start) / (1 << 30);  // GB/s

    printf("Device-to-device bandwidth (GPU %d -> GPU %d): %.2f GB/s\n", dev0, dev1, bandwidth);

    omp_target_free(src, dev0);
    omp_target_free(dst, dev1);

    return 0;
}
