```

cnh@dgx-06:~/nccl-test/nccl-tests$ mpirun -x LD_LIBRARY_PATH --hostfile myhosts -n 2  ./build/all_reduce_perf -b 8 -e 8192M -f 2 -g 4 -n 30
# nThread 1 nGpus 4 minBytes 8 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 30 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1467900 on     dgx-06 device  0 [0x1b] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 1467900 on     dgx-06 device  1 [0x43] NVIDIA H100 80GB HBM3
#  Rank  2 Group  0 Pid 1467900 on     dgx-06 device  2 [0x52] NVIDIA H100 80GB HBM3
#  Rank  3 Group  0 Pid 1467900 on     dgx-06 device  3 [0x61] NVIDIA H100 80GB HBM3
#  Rank  4 Group  0 Pid 2491224 on     dgx-07 device  0 [0x1b] NVIDIA H100 80GB HBM3
#  Rank  5 Group  0 Pid 2491224 on     dgx-07 device  1 [0x43] NVIDIA H100 80GB HBM3
#  Rank  6 Group  0 Pid 2491224 on     dgx-07 device  2 [0x52] NVIDIA H100 80GB HBM3
#  Rank  7 Group  0 Pid 2491224 on     dgx-07 device  3 [0x61] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    85.62    0.00    0.00      0    22.29    0.00    0.00      0
          16             4     float     sum      -1    24.58    0.00    0.00      0    24.55    0.00    0.00      0
          32             8     float     sum      -1    24.99    0.00    0.00      0    24.75    0.00    0.00      0
          64            16     float     sum      -1    24.51    0.00    0.00      0    24.87    0.00    0.00      0
         128            32     float     sum      -1    24.81    0.01    0.01      0    25.19    0.01    0.01      0
         256            64     float     sum      -1    35.05    0.01    0.01      0    24.74    0.01    0.02      0
         512           128     float     sum      -1    21.93    0.02    0.04      0    24.76    0.02    0.04      0
        1024           256     float     sum      -1    24.72    0.04    0.07      0    25.13    0.04    0.07      0
        2048           512     float     sum      -1    25.17    0.08    0.14      0    24.88    0.08    0.14      0
        4096          1024     float     sum      -1    22.46    0.18    0.32      0    22.00    0.19    0.33      0
        8192          2048     float     sum      -1    23.62    0.35    0.61      0    24.61    0.33    0.58      0
       16384          4096     float     sum      -1    295.4    0.06    0.10      0    27.08    0.61    1.06      0
       32768          8192     float     sum      -1    971.7    0.03    0.06      0    41.24    0.79    1.39      0
       65536         16384     float     sum      -1    50.70    1.29    2.26      0    50.11    1.31    2.29      0
      131072         32768     float     sum      -1    114.1    1.15    2.01      0    113.1    1.16    2.03      0
      262144         65536     float     sum      -1    210.7    1.24    2.18      0    209.1    1.25    2.19      0
      524288        131072     float     sum      -1    73.03    7.18   12.56      0    71.05    7.38   12.91      0
     1048576        262144     float     sum      -1    102.5   10.23   17.91      0    99.62   10.53   18.42      0
     2097152        524288     float     sum      -1    135.0   15.53   27.18      0    135.2   15.51   27.14      0
     4194304       1048576     float     sum      -1    262.8   15.96   27.93      0    250.5   16.75   29.30      0
     8388608       2097152     float     sum      -1    268.3   31.27   54.72      0    264.3   31.73   55.54      0
    16777216       4194304     float     sum      -1    561.5   29.88   52.29      0    510.3   32.88   57.53      0
    33554432       8388608     float     sum      -1    557.3   60.21  105.37      0    563.0   59.60  104.30      0
    67108864      16777216     float     sum      -1    853.8   78.60  137.54      0    857.0   78.31  137.04      0
   134217728      33554432     float     sum      -1   1563.0   85.87  150.28      0   1600.3   83.87  146.77      0
   268435456      67108864     float     sum      -1   1917.7  139.98  244.97      0   1878.2  142.92  250.12      0
   536870912     134217728     float     sum      -1   3278.4  163.76  286.58      0   3792.0  141.58  247.76      0
  1073741824     268435456     float     sum      -1   6741.1  159.28  278.74      0   6425.0  167.12  292.46      0
  2147483648     536870912     float     sum      -1    11533  186.21  325.86      0    12612  170.27  297.97      0
  4294967296    1073741824     float     sum      -1    23607  181.94  318.39      0    23506  182.72  319.76      0
  8589934592    2147483648     float     sum      -1    45789  187.60  328.29      0    45423  189.11  330.95      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 76.0414 
#



```
