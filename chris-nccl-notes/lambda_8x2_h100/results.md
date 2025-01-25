```
ubuntu@alpha-test-16way-node-001:~/cnh/nccl-bench/nccl-tests$ mpirun --hostfile myhosts -n 2 ./build/all_reduce_perf -b 8 -e 8192M -f 2 -g 8 -n 30
# nThread 1 nGpus 8 minBytes 8 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 30 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 148287 on alpha-test-16way-node-001 device  0 [0x63] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 148287 on alpha-test-16way-node-001 device  1 [0x6b] NVIDIA H100 80GB HBM3
#  Rank  2 Group  0 Pid 148287 on alpha-test-16way-node-001 device  2 [0x71] NVIDIA H100 80GB HBM3
#  Rank  3 Group  0 Pid 148287 on alpha-test-16way-node-001 device  3 [0x79] NVIDIA H100 80GB HBM3
#  Rank  4 Group  0 Pid 148287 on alpha-test-16way-node-001 device  4 [0x7f] NVIDIA H100 80GB HBM3
#  Rank  5 Group  0 Pid 148287 on alpha-test-16way-node-001 device  5 [0x87] NVIDIA H100 80GB HBM3
#  Rank  6 Group  0 Pid 148287 on alpha-test-16way-node-001 device  6 [0x8d] NVIDIA H100 80GB HBM3
#  Rank  7 Group  0 Pid 148287 on alpha-test-16way-node-001 device  7 [0x95] NVIDIA H100 80GB HBM3
#  Rank  8 Group  0 Pid 135721 on alpha-test-16way-node-002 device  0 [0x63] NVIDIA H100 80GB HBM3
#  Rank  9 Group  0 Pid 135721 on alpha-test-16way-node-002 device  1 [0x6b] NVIDIA H100 80GB HBM3
#  Rank 10 Group  0 Pid 135721 on alpha-test-16way-node-002 device  2 [0x71] NVIDIA H100 80GB HBM3
#  Rank 11 Group  0 Pid 135721 on alpha-test-16way-node-002 device  3 [0x79] NVIDIA H100 80GB HBM3
#  Rank 12 Group  0 Pid 135721 on alpha-test-16way-node-002 device  4 [0x7f] NVIDIA H100 80GB HBM3
#  Rank 13 Group  0 Pid 135721 on alpha-test-16way-node-002 device  5 [0x87] NVIDIA H100 80GB HBM3
#  Rank 14 Group  0 Pid 135721 on alpha-test-16way-node-002 device  6 [0x8d] NVIDIA H100 80GB HBM3
#  Rank 15 Group  0 Pid 135721 on alpha-test-16way-node-002 device  7 [0x95] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    62.35    0.00    0.00      0    61.02    0.00    0.00      0
          16             4     float     sum      -1    60.40    0.00    0.00      0    60.16    0.00    0.00      0
          32             8     float     sum      -1    61.01    0.00    0.00      0    60.47    0.00    0.00      0
          64            16     float     sum      -1    61.01    0.00    0.00      0    60.28    0.00    0.00      0
         128            32     float     sum      -1    61.26    0.00    0.00      0    60.46    0.00    0.00      0
         256            64     float     sum      -1    60.93    0.00    0.01      0    61.60    0.00    0.01      0
         512           128     float     sum      -1    60.88    0.01    0.02      0    60.51    0.01    0.02      0
        1024           256     float     sum      -1    60.24    0.02    0.03      0    62.06    0.02    0.03      0
        2048           512     float     sum      -1    60.17    0.03    0.06      0    60.50    0.03    0.06      0
        4096          1024     float     sum      -1    60.68    0.07    0.13      0    60.90    0.07    0.13      0
        8192          2048     float     sum      -1    61.27    0.13    0.25      0    61.25    0.13    0.25      0
       16384          4096     float     sum      -1    66.90    0.24    0.46      0    66.95    0.24    0.46      0
       32768          8192     float     sum      -1    75.12    0.44    0.82      0    72.78    0.45    0.84      0
       65536         16384     float     sum      -1    125.0    0.52    0.98      0    208.4    0.31    0.59      0
      131072         32768     float     sum      -1    160.7    0.82    1.53      0    170.7    0.77    1.44      0
      262144         65536     float     sum      -1    191.5    1.37    2.57      0    201.8    1.30    2.44      0
      524288        131072     float     sum      -1    557.0    0.94    1.76      0    479.8    1.09    2.05      0
     1048576        262144     float     sum      -1    463.1    2.26    4.25      0    516.6    2.03    3.81      0
     2097152        524288     float     sum      -1    487.4    4.30    8.07      0    486.2    4.31    8.09      0
     4194304       1048576     float     sum      -1    508.2    8.25   15.47      0    503.7    8.33   15.61      0
     8388608       2097152     float     sum      -1    560.9   14.96   28.04      0    584.8   14.34   26.90      0
    16777216       4194304     float     sum      -1    744.1   22.55   42.27      0    725.1   23.14   43.38      0
    33554432       8388608     float     sum      -1   1094.6   30.66   57.48      0   1070.7   31.34   58.76      0
    67108864      16777216     float     sum      -1   1351.1   49.67   93.13      0   1195.0   56.16  105.30      0
   134217728      33554432     float     sum      -1   2015.4   66.60  124.87      0   1611.2   83.30  156.19      0
   268435456      67108864     float     sum      -1   3018.4   88.93  166.75      0   3359.2   79.91  149.83      0
   536870912     134217728     float     sum      -1   5760.4   93.20  174.75      0   5726.7   93.75  175.78      0
  1073741824     268435456     float     sum      -1    10381  103.43  193.93      0    10157  105.71  198.21      0
  2147483648     536870912     float     sum      -1    19099  112.44  210.82      0    19229  111.68  209.40      0
  4294967296    1073741824     float     sum      -1    37372  114.92  215.48      0    37893  113.34  212.52      0
  8589934592    2147483648     float     sum      -1    73372  117.07  219.51      0    73006  117.66  220.61      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 50.906 
#

```
