#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define d1 6000
#define d2 6000
#define d3 6000

float** createDynamicArray(int rows, int cols) {
	float** array = malloc(rows * sizeof(float*));
	int i;
	for (i = 0; i < rows; i++) {
		array[i] = malloc(cols * sizeof(float));
	}
	return array; // Return the pointer to the dynamic array
}

int main() {
	float** m1 = createDynamicArray(d1, d2);
	float** m2 = createDynamicArray(d2, d3);
	float** ans = createDynamicArray(d1, d3);
	float lower_bound = 1;
	float upper_bound = 10;
	int i, j, k;
#pragma omp parallel private(j)
	{
		unsigned int seed = time(NULL) + omp_get_thread_num();
		srand(seed);
		#pragma omp for
		for (i = 0;i < d1;i++) {
			for (j = 0;j < d2;j++) {
				m1[i][j] = rand() % 101;
			}
		}
		#pragma omp for
		for (i = 0;i < d2;i++) {
			for (j = 0;j < d3;j++) {
				m2[i][j] = rand();
			}
		}
	}
	double start = omp_get_wtime();
#pragma omp parallel private(j, k)
	{
		#pragma omp for
		for (i = 0;i < d1;i++) {
			for (k = 0;k < d3;k++) {
				ans[i][k] = 0;
				for (j = 0;j < d2;j++) {
					ans[i][k] += m1[i][j] * m2[j][k];
				}
			}
		}
	}
	double end = omp_get_wtime();
	printf("%d threads took %lf time to matrix multiply\n", omp_get_num_threads(), end - start);
	return 0;
}
