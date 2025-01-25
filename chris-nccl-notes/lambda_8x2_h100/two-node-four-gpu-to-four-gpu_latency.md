```
ubuntu@alpha-test-16way-node-001:~/cnh/nccl-bench/nccl-tests$ mpirun --hostfile myhosts -n 2 ./build/all_reduce_perf -b 8 -e 16384M -f 2 -g 4 -n 30
# nThread 1 nGpus 4 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 30 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid 154970 on alpha-test-16way-node-001 device  0 [0x63] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 154970 on alpha-test-16way-node-001 device  1 [0x6b] NVIDIA H100 80GB HBM3
#  Rank  2 Group  0 Pid 154970 on alpha-test-16way-node-001 device  2 [0x71] NVIDIA H100 80GB HBM3
#  Rank  3 Group  0 Pid 154970 on alpha-test-16way-node-001 device  3 [0x79] NVIDIA H100 80GB HBM3
#  Rank  4 Group  0 Pid 140440 on alpha-test-16way-node-002 device  0 [0x63] NVIDIA H100 80GB HBM3
#  Rank  5 Group  0 Pid 140440 on alpha-test-16way-node-002 device  1 [0x6b] NVIDIA H100 80GB HBM3
#  Rank  6 Group  0 Pid 140440 on alpha-test-16way-node-002 device  2 [0x71] NVIDIA H100 80GB HBM3
#  Rank  7 Group  0 Pid 140440 on alpha-test-16way-node-002 device  3 [0x79] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    32.07    0.00    0.00      0    33.05    0.00    0.00      0
          16             4     float     sum      -1    32.69    0.00    0.00      0    33.40    0.00    0.00      0
          32             8     float     sum      -1    33.43    0.00    0.00      0    32.80    0.00    0.00      0
          64            16     float     sum      -1    36.55    0.00    0.00      0    32.70    0.00    0.00      0
         128            32     float     sum      -1    32.86    0.00    0.01      0    39.89    0.00    0.01      0
         256            64     float     sum      -1    38.60    0.01    0.01      0    39.77    0.01    0.01      0
         512           128     float     sum      -1    33.02    0.02    0.03      0    33.09    0.02    0.03      0
        1024           256     float     sum      -1    32.45    0.03    0.06      0    32.38    0.03    0.06      0
        2048           512     float     sum      -1    32.32    0.06    0.11      0    32.58    0.06    0.11      0
        4096          1024     float     sum      -1    32.78    0.12    0.22      0    32.65    0.13    0.22      0
        8192          2048     float     sum      -1    33.82    0.24    0.42      0    32.78    0.25    0.44      0
       16384          4096     float     sum      -1    34.95    0.47    0.82      0    36.10    0.45    0.79      0
       32768          8192     float     sum      -1    44.81    0.73    1.28      0    43.15    0.76    1.33      0
       65536         16384     float     sum      -1    79.77    0.82    1.44      0    73.88    0.89    1.55      0
      131072         32768     float     sum      -1    141.9    0.92    1.62      0    131.7    1.00    1.74      0
      262144         65536     float     sum      -1    207.5    1.26    2.21      0    205.5    1.28    2.23      0
      524288        131072     float     sum      -1    146.7    3.57    6.25      0    133.3    3.93    6.88      0
     1048576        262144     float     sum      -1    171.5    6.11   10.70      0    172.5    6.08   10.64      0
     2097152        524288     float     sum      -1    644.3    3.25    5.70      0    478.4    4.38    7.67      0
     4194304       1048576     float     sum      -1    253.1   16.57   29.00      0    245.1   17.11   29.95      0
     8388608       2097152     float     sum      -1    325.7   25.76   45.08      0    320.6   26.16   45.78      0
    16777216       4194304     float     sum      -1    445.4   37.67   65.92      0    449.1   37.36   65.38      0
    33554432       8388608     float     sum      -1    520.6   64.46  112.80      0    566.0   59.28  103.75      0
    67108864      16777216     float     sum      -1    754.3   88.97  155.69      0    738.5   90.87  159.02      0
   134217728      33554432     float     sum      -1   1300.8  103.18  180.57      0   1286.8  104.30  182.53      0
   268435456      67108864     float     sum      -1   2352.7  114.10  199.67      0   2371.0  113.22  198.13      0
   536870912     134217728     float     sum      -1   4334.2  123.87  216.77      0   4649.4  115.47  202.08      0
  1073741824     268435456     float     sum      -1   8687.1  123.60  216.30      0   8719.1  123.15  215.51      0
  2147483648     536870912     float     sum      -1    16195  132.60  232.06      0    16921  126.91  222.09      0
  4294967296    1073741824     float     sum      -1    31973  134.33  235.08      0    32555  131.93  230.88      0
  8589934592    2147483648     float     sum      -1    64288  133.62  233.83      0    64110  133.99  234.48      0
 17179869184    4294967296     float     sum      -1   127660  134.58  235.51      0   128172  134.04  234.57      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 67.9217 
#

```
