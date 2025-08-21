# Example run and output

```
[cnh@node3400 fio_testing]$ ./fio_fs_quick.sh /scratch
--- SEQ-READ (read, 1M, iodepth=32) ---
  read: IOPS=43.6k, BW=42.6GiB/s (45.7GB/s)(1278GiB/30006msec)
   READ: bw=42.6GiB/s (45.7GB/s), 42.6GiB/s-42.6GiB/s (45.7GB/s-45.7GB/s), io=1278GiB (1373GB), run=30006-30006msec
--- SEQ-WRITE (write, 1M, iodepth=32) ---
  write: IOPS=25.8k, BW=25.1GiB/s (27.0GB/s)(755GiB/30010msec); 0 zone resets
  WRITE: bw=25.1GiB/s (27.0GB/s), 25.1GiB/s-25.1GiB/s (27.0GB/s-27.0GB/s), io=755GiB (810GB), run=30010-30010msec
--- RAND-READ (randread, 4k, iodepth=128) ---
  read: IOPS=1279k, BW=4995MiB/s (5238MB/s)(146GiB/30002msec)
   READ: bw=4995MiB/s (5238MB/s), 4995MiB/s-4995MiB/s (5238MB/s-5238MB/s), io=146GiB (157GB), run=30002-30002msec
--- RAND-WRITE (randwrite, 4k, iodepth=128) ---
  write: IOPS=997k, BW=3894MiB/s (4083MB/s)(114GiB/30001msec); 0 zone resets
  WRITE: bw=3894MiB/s (4083MB/s), 3894MiB/s-3894MiB/s (4083MB/s-4083MB/s), io=114GiB (123GB), run=30001-30001msec
Results saved to fio_fs_results.txt
[cnh@node3400 fio_testing]$ 

