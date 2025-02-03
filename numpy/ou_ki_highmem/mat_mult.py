import time
import numpy as np

n = 20000 # 10000

arr1 = np.random.rand(n,n)
arr2 = np.random.rand(n,n)
t0 = time.time()
#arr_result = np.multiply(arr1, arr2)  # serial
arr_result = np.matmul(arr1, arr2)     # multithreads
#print(arr_result)

t1 = time.time()
print("Time to compute:", t1 - t0)

