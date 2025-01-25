```

cnh@dgx-06:~/nccl-test/nccl-tests$ mpirun -x LD_LIBRARY_PATH --hostfile myhosts -n 2  ./build/all_reduce_perf -b 8 -e 8192M -f 2 -g 2 -n 30
# nThread 1 nGpus 2 minBytes 8 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 30 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1462688 on     dgx-06 device  0 [0x1b] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 1462688 on     dgx-06 device  1 [0x43] NVIDIA H100 80GB HBM3
#  Rank  2 Group  0 Pid 2485837 on     dgx-07 device  0 [0x1b] NVIDIA H100 80GB HBM3
#  Rank  3 Group  0 Pid 2485837 on     dgx-07 device  1 [0x43] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    16.03    0.00    0.00      0    15.73    0.00    0.00      0
          16             4     float     sum      -1    15.67    0.00    0.00      0    15.84    0.00    0.00      0
          32             8     float     sum      -1    15.90    0.00    0.00      0    15.72    0.00    0.00      0
          64            16     float     sum      -1    15.89    0.00    0.01      0    15.86    0.00    0.01      0
         128            32     float     sum      -1    16.14    0.01    0.01      0    16.33    0.01    0.01      0
         256            64     float     sum      -1    16.51    0.02    0.02      0    16.14    0.02    0.02      0
         512           128     float     sum      -1    16.77    0.03    0.05      0    16.65    0.03    0.05      0
        1024           256     float     sum      -1    17.51    0.06    0.09      0    17.40    0.06    0.09      0
        2048           512     float     sum      -1    18.27    0.11    0.17      0    18.37    0.11    0.17      0
        4096          1024     float     sum      -1    19.19    0.21    0.32      0    18.84    0.22    0.33      0
        8192          2048     float     sum      -1    20.21    0.41    0.61      0    19.61    0.42    0.63      0
       16384          4096     float     sum      -1    291.4    0.06    0.08      0    24.76    0.66    0.99      0
       32768          8192     float     sum      -1    29.22    1.12    1.68      0    29.55    1.11    1.66      0
       65536         16384     float     sum      -1    39.91    1.64    2.46      0    40.58    1.61    2.42      0
      131072         32768     float     sum      -1    74.75    1.75    2.63      0    72.46    1.81    2.71      0
      262144         65536     float     sum      -1    93.30    2.81    4.21      0    93.15    2.81    4.22      0
      524288        131072     float     sum      -1    63.30    8.28   12.42      0    67.97    7.71   11.57      0
     1048576        262144     float     sum      -1    79.10   13.26   19.88      0    67.46   15.54   23.31      0
     2097152        524288     float     sum      -1    77.74   26.98   40.47      0    77.81   26.95   40.43      0
     4194304       1048576     float     sum      -1    101.2   41.43   62.14      0    100.2   41.85   62.78      0
     8388608       2097152     float     sum      -1    172.0   48.76   73.14      0    171.3   48.96   73.44      0
    16777216       4194304     float     sum      -1    317.1   52.91   79.36      0    322.7   52.00   77.99      0
    33554432       8388608     float     sum      -1    588.5   57.01   85.52      0    588.7   57.00   85.50      0
    67108864      16777216     float     sum      -1   1101.0   60.95   91.43      0   1101.6   60.92   91.38      0
   134217728      33554432     float     sum      -1   1479.2   90.74  136.10      0   1480.2   90.67  136.01      0
   268435456      67108864     float     sum      -1   2838.5   94.57  141.85      0   2839.3   94.54  141.81      0
   536870912     134217728     float     sum      -1   5561.2   96.54  144.81      0   5561.2   96.54  144.81      0
  1073741824     268435456     float     sum      -1    10996   97.65  146.47      0    11010   97.52  146.28      0
  2147483648     536870912     float     sum      -1    21872   98.18  147.27      0    21933   97.91  146.87      0
  4294967296    1073741824     float     sum      -1    44024   97.56  146.34      0    43866   97.91  146.87      0
  8589934592    2147483648     float     sum      -1    87072   98.65  147.98      0    87098   98.62  147.94      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 48.0299 
#


```
