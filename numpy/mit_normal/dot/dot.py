import numpy as np
import time
n = 20000

arr1 = np.random.rand(n, n)
arr2 = np.random.rand(n, n)

cur_time = time.time()
res = np.dot(arr1, arr2)
print(time.time() - cur_time)
