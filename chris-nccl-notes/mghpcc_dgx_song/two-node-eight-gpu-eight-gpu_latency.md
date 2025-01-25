```

cnh@dgx-06:~/nccl-test/nccl-tests$ mpirun -x LD_LIBRARY_PATH --hostfile myhosts -n 2  ./build/all_reduce_perf -b 8 -e 16384M -f 2 -g 8 -n 30
# nThread 1 nGpus 8 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 30 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 1471953 on     dgx-06 device  0 [0x1b] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 1471953 on     dgx-06 device  1 [0x43] NVIDIA H100 80GB HBM3
#  Rank  2 Group  0 Pid 1471953 on     dgx-06 device  2 [0x52] NVIDIA H100 80GB HBM3
#  Rank  3 Group  0 Pid 1471953 on     dgx-06 device  3 [0x61] NVIDIA H100 80GB HBM3
#  Rank  4 Group  0 Pid 1471953 on     dgx-06 device  4 [0x9d] NVIDIA H100 80GB HBM3
#  Rank  5 Group  0 Pid 1471953 on     dgx-06 device  5 [0xc3] NVIDIA H100 80GB HBM3
#  Rank  6 Group  0 Pid 1471953 on     dgx-06 device  6 [0xd1] NVIDIA H100 80GB HBM3
#  Rank  7 Group  0 Pid 1471953 on     dgx-06 device  7 [0xdf] NVIDIA H100 80GB HBM3
#  Rank  8 Group  0 Pid 2495322 on     dgx-07 device  0 [0x1b] NVIDIA H100 80GB HBM3
#  Rank  9 Group  0 Pid 2495322 on     dgx-07 device  1 [0x43] NVIDIA H100 80GB HBM3
#  Rank 10 Group  0 Pid 2495322 on     dgx-07 device  2 [0x52] NVIDIA H100 80GB HBM3
#  Rank 11 Group  0 Pid 2495322 on     dgx-07 device  3 [0x61] NVIDIA H100 80GB HBM3
#  Rank 12 Group  0 Pid 2495322 on     dgx-07 device  4 [0x9d] NVIDIA H100 80GB HBM3
#  Rank 13 Group  0 Pid 2495322 on     dgx-07 device  5 [0xc3] NVIDIA H100 80GB HBM3
#  Rank 14 Group  0 Pid 2495322 on     dgx-07 device  6 [0xd1] NVIDIA H100 80GB HBM3
#  Rank 15 Group  0 Pid 2495322 on     dgx-07 device  7 [0xdf] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    172.7    0.00    0.00      0    40.22    0.00    0.00      0
          16             4     float     sum      -1    39.09    0.00    0.00      0    40.11    0.00    0.00      0
          32             8     float     sum      -1    39.07    0.00    0.00      0    38.76    0.00    0.00      0
          64            16     float     sum      -1    38.80    0.00    0.00      0    39.43    0.00    0.00      0
         128            32     float     sum      -1    39.71    0.00    0.01      0    40.23    0.00    0.01      0
         256            64     float     sum      -1    52.25    0.00    0.01      0    39.54    0.01    0.01      0
         512           128     float     sum      -1    39.28    0.01    0.02      0    39.60    0.01    0.02      0
        1024           256     float     sum      -1    39.56    0.03    0.05      0    40.03    0.03    0.05      0
        2048           512     float     sum      -1    40.22    0.05    0.10      0    39.95    0.05    0.10      0
        4096          1024     float     sum      -1    40.82    0.10    0.19      0    40.17    0.10    0.19      0
        8192          2048     float     sum      -1    296.5    0.03    0.05      0    44.98    0.18    0.34      0
       16384          4096     float     sum      -1    966.6    0.02    0.03      0    49.79    0.33    0.62      0
       32768          8192     float     sum      -1   1112.2    0.03    0.06      0   2068.8    0.02    0.03      0
       65536         16384     float     sum      -1   1656.1    0.04    0.07      0   1651.7    0.04    0.07      0
      131072         32768     float     sum      -1   1660.1    0.08    0.15      0    783.8    0.17    0.31      0
      262144         65536     float     sum      -1    580.3    0.45    0.85      0    328.8    0.80    1.49      0
      524288        131072     float     sum      -1    744.8    0.70    1.32      0    742.9    0.71    1.32      0
     1048576        262144     float     sum      -1    764.0    1.37    2.57      0    760.0    1.38    2.59      0
     2097152        524288     float     sum      -1    844.6    2.48    4.66      0   1517.2    1.38    2.59      0
     4194304       1048576     float     sum      -1   1131.0    3.71    6.95      0   1172.0    3.58    6.71      0
     8388608       2097152     float     sum      -1   2022.1    4.15    7.78      0   2282.8    3.67    6.89      0
    16777216       4194304     float     sum      -1   1373.1   12.22   22.91      0   1472.1   11.40   21.37      0
    33554432       8388608     float     sum      -1   1568.7   21.39   40.11      0   2170.2   15.46   28.99      0
    67108864      16777216     float     sum      -1   2219.0   30.24   56.71      0   2523.1   26.60   49.87      0
   134217728      33554432     float     sum      -1   1615.8   83.07  155.75      0   1894.2   70.86  132.86      0
   268435456      67108864     float     sum      -1   3231.0   83.08  155.78      0   2927.8   91.69  171.91      0
   536870912     134217728     float     sum      -1   3345.7  160.47  300.87      0   3417.6  157.09  294.55      0
  1073741824     268435456     float     sum      -1   5961.9  180.10  337.69      0   5424.4  197.95  371.15      0
  2147483648     536870912     float     sum      -1   9270.2  231.66  434.35      0   9326.8  230.25  431.71      0
  4294967296    1073741824     float     sum      -1    17662  243.18  455.95      0    17583  244.27  458.01      0
  8589934592    2147483648     float     sum      -1    34409  249.65  468.09      0    34955  245.74  460.77      0
 17179869184    4294967296     float     sum      -1    68195  251.92  472.35      0    68483  250.86  470.37      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 91.2553 
#



```
