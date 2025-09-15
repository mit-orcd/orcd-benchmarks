#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 5000  // Size of the matrices

void initialize_matrices(double A[N][N], double B[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10; // Random values between 0-9
            B[i][j] = rand() % 10; // Random values between 0-9
        }
    }
}

void print_matrix(double matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

void matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    double A[N][N], B[N][N], C[N][N];

    // Initialize matrices
    initialize_matrices(A, B);
    double start = omp_get_wtime();
    // Perform matrix multiplication
    matrix_multiply(A, B, C);
    double end = omp_get_wtime();

    printf("Time to compute: %f \n", end - start);

    // Optionally print the result matrix
    // print_matrix(C);

    return 0;
}

