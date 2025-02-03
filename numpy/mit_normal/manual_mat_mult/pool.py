from multiprocessing import Pool, cpu_count
import numpy as np
import time

# Matrix size and number of matrices to multiply
n = 2000  # matrix dimensions (e.g., 1000x1000)
m = 100    # number of matrix pairs

# Generate random matrices
mat1 = [np.random.rand(n, n) for _ in range(m)]
mat2 = [np.random.rand(n, n) for _ in range(m)]
pairs = list(zip(mat1, mat2))

# Matrix multiplication function
def mat_mul(matrices):
    m1, m2 = matrices
    return np.matmul(m1, m2)

# Function to run the computation with a specified number of processes
def run_multiprocessing(num_processes):
    print(f"Running with {num_processes} processes...")

    # Start timing
    start_time = time.time()

    # Create a pool of workers and run the matrix multiplication in parallel
    with Pool(processes=num_processes) as pool:
        pool.map(mat_mul, pairs)

    # End timing
    end_time = time.time()
    duration = end_time - start_time

    print(f"Time taken with {num_processes} processes: {duration:.2f} seconds\n")
    return duration

if __name__ == "__main__":
    # Run with different numbers of processes
    sequential_time = run_multiprocessing(1)  # Sequential (1 process) for baseline

    # Test with a few parallel configurations
    for num_processes in [2, 4, 8, 16, 24, 36, 48, 72, 96]:
        parallel_time = run_multiprocessing(num_processes)
        speedup = sequential_time / parallel_time
        print(f"Speedup with {num_processes} processes: {speedup:.2f}x\n")
    print(cpu_count())
