from multiprocessing import Process, Pool, cpu_count
import numpy as np
import time
import random

n = 2000
mat1 = np.random.rand(n, n)
mat2 = np.random.rand(n, n)

def mat_mul(i):
  ans_row = np.zeros(n)

  for j in range(n):
    ans_row[j] = sum(mat1[i][k] * mat2[k][j] for k in range(n))
  return ans_row

start = time.time()
with Pool() as pool:
    ans = pool.map(mat_mul, range(n))
print(time.time() - start)
