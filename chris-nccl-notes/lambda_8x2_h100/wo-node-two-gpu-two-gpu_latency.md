```

ubuntu@alpha-test-16way-node-001:~/cnh/nccl-bench/nccl-tests$ mpirun --hostfile myhosts -n 2 ./build/all_reduce_perf -b 8 -e 16384M -f 2 -g 2 -n 30
# nThread 1 nGpus 2 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 30 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 154246 on alpha-test-16way-node-001 device  0 [0x63] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 154246 on alpha-test-16way-node-001 device  1 [0x6b] NVIDIA H100 80GB HBM3
#  Rank  2 Group  0 Pid 139703 on alpha-test-16way-node-002 device  0 [0x63] NVIDIA H100 80GB HBM3
#  Rank  3 Group  0 Pid 139703 on alpha-test-16way-node-002 device  1 [0x6b] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    17.61    0.00    0.00      0    17.46    0.00    0.00      0
          16             4     float     sum      -1    17.80    0.00    0.00      0    17.60    0.00    0.00      0
          32             8     float     sum      -1    42.56    0.00    0.00      0    18.77    0.00    0.00      0
          64            16     float     sum      -1    24.98    0.00    0.00      0    20.70    0.00    0.00      0
         128            32     float     sum      -1    21.22    0.01    0.01      0    20.80    0.01    0.01      0
         256            64     float     sum      -1    21.40    0.01    0.02      0    21.08    0.01    0.02      0
         512           128     float     sum      -1    21.58    0.02    0.04      0    21.29    0.02    0.04      0
        1024           256     float     sum      -1    25.26    0.04    0.06      0    21.71    0.05    0.07      0
        2048           512     float     sum      -1    22.07    0.09    0.14      0    21.89    0.09    0.14      0
        4096          1024     float     sum      -1    22.89    0.18    0.27      0    22.52    0.18    0.27      0
        8192          2048     float     sum      -1    24.23    0.34    0.51      0    23.66    0.35    0.52      0
       16384          4096     float     sum      -1    25.36    0.65    0.97      0    25.81    0.63    0.95      0
       32768          8192     float     sum      -1    37.48    0.87    1.31      0    36.63    0.89    1.34      0
       65536         16384     float     sum      -1    50.32    1.30    1.95      0    48.74    1.34    2.02      0
      131072         32768     float     sum      -1    90.31    1.45    2.18      0    86.42    1.52    2.28      0
      262144         65536     float     sum      -1    178.3    1.47    2.20      0    171.8    1.53    2.29      0
      524288        131072     float     sum      -1    363.9    1.44    2.16      0    355.9    1.47    2.21      0
     1048576        262144     float     sum      -1    250.2    4.19    6.29      0    249.8    4.20    6.30      0
     2097152        524288     float     sum      -1    245.3    8.55   12.82      0    242.5    8.65   12.97      0
     4194304       1048576     float     sum      -1    245.7   17.07   25.61      0    234.5   17.89   26.83      0
     8388608       2097152     float     sum      -1    268.7   31.22   46.83      0    265.7   31.58   47.36      0
    16777216       4194304     float     sum      -1    371.7   45.13   67.70      0    368.8   45.49   68.23      0
    33554432       8388608     float     sum      -1    650.5   51.58   77.37      0    658.6   50.95   76.43      0
    67108864      16777216     float     sum      -1   1121.4   59.84   89.76      0   1114.9   60.20   90.29      0
   134217728      33554432     float     sum      -1   1506.2   89.11  133.67      0   1502.2   89.35  134.02      0
   268435456      67108864     float     sum      -1   4253.1   63.12   94.67      0   4252.6   63.12   94.68      0
   536870912     134217728     float     sum      -1   5598.8   95.89  143.84      0   5712.3   93.99  140.98      0
  1073741824     268435456     float     sum      -1    11104   96.69  145.04      0    11054   97.13  145.70      0
  2147483648     536870912     float     sum      -1    22053   97.38  146.07      0    22140   96.99  145.49      0
  4294967296    1073741824     float     sum      -1    43878   97.88  146.83      0    43808   98.04  147.06      0
  8589934592    2147483648     float     sum      -1    87985   97.63  146.44      0    87554   98.11  147.16      0
 17179869184    4294967296     float     sum      -1   175040   98.15  147.22      0   174900   98.23  147.34      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 45.0781 
#

```
