import time
import numpy as np

n = 20000 # 10000
arr1 = np.random.rand(n,n)
arr2 = np.random.rand(n,n)

#arr_result = np.multiply(arr1, arr2)  # serial
time1 = time.time()
arr_result = np.matmul(arr1, arr2)     # multithreads
time2 = time.time()
print(f"{time2-time1} seconds to matrix multiply")
#print(arr_result)

