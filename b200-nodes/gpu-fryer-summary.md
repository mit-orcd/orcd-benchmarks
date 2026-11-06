# GPU Fryer Results

**Files:** `out-gpu-fryer/gpu-fryer-<node>-<date>.out`, newest run per node (12 nodes, 8× NVIDIA B200 each)
Each file contains three back-to-back runs: **FP32 → BF16 → FP8**, 300 s per precision.

---

## Summary Table (per-GPU mean, in TFLOPS)

The B200 row is the mean across all 96 GPUs on the 12 nodes.

| GPU / Metric | **GPU memory** | **FP32 TFLOPS** | **BF16 TFLOPS** | **FP8 TFLOPS** |
|---|---|---|---|---|
| **B200** (12 nodes) | ~160 GB | **749** | **1,457** | **4,010** |
| **H200** (ref) | 144 GB | **368** | **713** | **1,468** |
| **L40S** (ref) | 48 GB | **98** | **198** | not supported |
| **B200 / H200** | — | **2.04×** | **2.04×** | **2.73×** |

Precision ratios: B200 → 1 : 1.94 : 5.35 (FP32 : BF16 : FP8); H200 → 1 : 1.94 : 3.99; L40S → 1 : 2.02 (FP32 : BF16, no FP8 Tensor Core).

---

## Per-Node Summary — B200

| Node | Run date | FP32 | BF16 | FP8 | Peak temp | Result |
|---|---|---:|---:|---:|---:|---|
| node5500 | 2026-08-06 | 744 | 1,457 | 3,990 | 75°C | healthy, no throttling |
| node5501 | 2026-08-06 | 748 | 1,457 | 4,001 | 75°C | healthy, no throttling |
| node5502 | 2026-08-06 | 760 | 1,437 | 4,062 | 75°C | healthy, no throttling |
| node5600-c1 | 2026-08-24 | 740 | 1,451 | 3,949 | 73°C | healthy, no throttling |
| node5601-c1 | 2026-08-24 | 747 | 1,451 | 4,006 | 73°C | healthy, no throttling |
| node5602-c1 | 2026-08-24 | 745 | 1,461 | 4,011 | 74°C | healthy, no throttling |
| node5700-c1 | 2026-10-27 | 749 | 1,462 | 4,018 | 73°C | healthy, no throttling |
| node5701-c1 | 2026-10-27 | 749 | 1,462 | 4,014 | 73°C | healthy, no throttling |
| node5702-c1 | 2026-08-24 | 752 | 1,459 | 3,981 | 75°C | healthy, no throttling |
| node5800-c1 | 2026-08-24 | 752 | 1,465 | 4,028 | 74°C | healthy, no throttling |
| node5801-c1 | 2026-08-24 | 751 | 1,457 | 4,032 | 75°C | healthy, no throttling |
| node5802-c1 | 2026-08-24 | 749 | 1,461 | 4,020 | 74°C | healthy, no throttling |
| **Mean, 12 nodes** | — | **749** | **1,457** | **4,010** | — | — |

Per-node means stay within 1.5% of the 12-node mean on every precision.

---

## Per-GPU Converged TFLOPS

### B200 — node5500

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 740.9 / 813.6 | 1,480.4 / 1,585.5 | 4,049.2 / 4,362.9 | 61°C |
| #1 | 755.1 / 830.1 | 1,470.5 / 1,575.6 | 4,064.6 / 4,369.5 | 64°C |
| #2 | 738.5 / 810.3 | 1,440.6 / 1,544.8 | 3,921.7 / 4,216.6 | 75°C |
| #3 | 736.1 / 809.2 | 1,439.6 / 1,541.5 | 3,940.8 / 4,237.5 | 73°C |
| #4 | 752.7 / 829.0 | 1,470.3 / 1,574.5 | 4,002.6 / 4,303.5 | 64°C |
| #5 | 747.9 / 821.3 | 1,458.6 / 1,563.5 | 4,035.6 / 4,339.8 | 61°C |
| #6 | 740.8 / 813.6 | 1,450.2 / 1,554.7 | 3,920.9 / 4,214.4 | 73°C |
| #7 | 740.9 / 812.5 | 1,444.2 / 1,548.1 | 3,988.4 / 4,290.3 | 75°C |
| **8-GPU mean** | **744 / 817** | **1,457 / 1,561** | **3,990 / 4,292** | — |

### B200 — node5501

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 759.2 / 776.3 | 1,476.6 / 1,486.5 | 4,059.5 / 4,081.4 | 61°C |
| #1 | 747.3 / 763.1 | 1,452.8 / 1,463.4 | 4,012.1 / 4,048.4 | 64°C |
| #2 | 742.4 / 759.8 | 1,446.1 / 1,462.4 | 3,993.2 / 4,023.1 | 75°C |
| #3 | 754.4 / 775.2 | 1,469.3 / 1,476.6 | 4,026.3 / 4,068.2 | 70°C |
| #4 | 735.4 / 754.3 | 1,429.2 / 1,437.1 | 3,974.8 / 3,992.3 | 66°C |
| #5 | 749.6 / 767.5 | 1,460.4 / 1,471.1 | 4,045.5 / 4,066.0 | 62°C |
| #6 | 751.9 / 773.0 | 1,471.7 / 1,485.4 | 3,936.2 / 4,002.2 | 74°C |
| #7 | 740.0 / 766.4 | 1,446.1 / 1,455.8 | 3,960.1 / 3,989.0 | 75°C |
| **8-GPU mean** | **748 / 767** | **1,457 / 1,467** | **4,001 / 4,034** | — |

### B200 — node5502

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 759.7 / 804.8 | 1,445.1 / 1,603.1 | 4,087.0 / 4,444.2 | 61°C |
| #1 | 769.3 / 822.4 | 1,441.1 / 1,597.6 | 4,113.7 / 4,456.3 | 63°C |
| #2 | 755.0 / 802.6 | 1,434.7 / 1,588.8 | 4,048.0 / 4,381.6 | 73°C |
| #3 | 764.5 / 811.4 | 1,447.6 / 1,606.4 | 4,092.0 / 4,434.3 | 70°C |
| #4 | 766.8 / 821.3 | 1,444.8 / 1,611.9 | 4,063.4 / 4,433.2 | 65°C |
| #5 | 757.4 / 801.5 | 1,432.0 / 1,586.6 | 4,011.1 / 4,404.6 | 61°C |
| #6 | 755.1 / 805.9 | 1,424.9 / 1,577.8 | 4,058.6 / 4,403.5 | 71°C |
| #7 | 754.5 / 810.3 | 1,425.6 / 1,578.9 | 4,026.0 / 4,386.0 | 75°C |
| **8-GPU mean** | **760 / 810** | **1,437 / 1,594** | **4,062 / 4,418** | — |

### B200 — node5600-c1

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 754.2 / 766.4 | 1,473.1 / 1,485.4 | 4,045.5 / 4,092.4 | 60°C |
| #1 | 747.9 / 773.0 | 1,453.3 / 1,466.7 | 3,973.6 / 4,068.2 | 62°C |
| #2 | 730.9 / 752.1 | 1,428.4 / 1,447.0 | 3,931.6 / 3,990.1 | 73°C |
| #3 | 737.1 / 767.5 | 1,442.9 / 1,459.1 | 3,955.4 / 4,007.7 | 73°C |
| #4 | 739.3 / 763.1 | 1,457.7 / 1,470.0 | 3,935.8 / 4,015.4 | 65°C |
| #5 | 744.3 / 763.1 | 1,453.2 / 1,466.7 | 3,964.3 / 4,037.4 | 61°C |
| #6 | 739.6 / 765.3 | 1,445.2 / 1,462.4 | 3,898.5 / 3,962.6 | 72°C |
| #7 | 727.9 / 767.5 | 1,456.6 / 1,478.8 | 3,889.1 / 3,941.7 | 73°C |
| **8-GPU mean** | **740 / 765** | **1,451 / 1,467** | **3,949 / 4,014** | — |

### B200 — node5601-c1

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 744.0 / 763.1 | 1,435.3 / 1,445.9 | 4,039.2 / 4,066.0 | 60°C |
| #1 | 755.6 / 774.1 | 1,465.5 / 1,479.9 | 4,064.8 / 4,093.5 | 62°C |
| #2 | 750.8 / 768.6 | 1,464.3 / 1,478.8 | 3,938.5 / 3,963.7 | 73°C |
| #3 | 746.6 / 776.3 | 1,456.1 / 1,472.2 | 3,948.2 / 3,994.5 | 72°C |
| #4 | 749.0 / 767.5 | 1,454.7 / 1,466.7 | 4,043.4 / 4,062.7 | 63°C |
| #5 | 751.5 / 767.5 | 1,451.7 / 1,471.1 | 4,054.2 / 4,084.7 | 61°C |
| #6 | 739.5 / 769.7 | 1,440.4 / 1,462.4 | 3,980.3 / 4,007.7 | 72°C |
| #7 | 742.0 / 764.2 | 1,442.4 / 1,460.2 | 3,978.0 / 4,020.9 | 73°C |
| **8-GPU mean** | **747 / 769** | **1,451 / 1,467** | **4,006 / 4,037** | — |

### B200 — node5602-c1

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 755.5 / 801.5 | 1,478.8 / 1,516.2 | 4,073.8 / 4,537.7 | 61°C |
| #1 | 757.8 / 799.3 | 1,484.7 / 1,521.7 | 4,083.2 / 4,530.0 | 61°C |
| #2 | 724.8 / 773.0 | 1,423.4 / 1,463.4 | 3,939.0 / 4,366.2 | 72°C |
| #3 | 743.7 / 782.9 | 1,468.4 / 1,506.3 | 3,962.1 / 4,399.1 | 72°C |
| #4 | 746.1 / 779.6 | 1,458.9 / 1,497.5 | 4,062.7 / 4,509.1 | 64°C |
| #5 | 748.9 / 790.5 | 1,454.3 / 1,492.0 | 4,060.6 / 4,514.6 | 61°C |
| #6 | 739.4 / 779.6 | 1,463.3 / 1,500.8 | 3,935.3 / 4,366.2 | 72°C |
| #7 | 741.9 / 778.5 | 1,455.4 / 1,492.0 | 3,972.7 / 4,409.0 | 74°C |
| **8-GPU mean** | **745 / 786** | **1,461 / 1,499** | **4,011 / 4,454** | — |

### B200 — node5700-c1

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 758.5 / 774.1 | 1,477.3 / 1,488.7 | 4,088.3 / 4,099.0 | 61°C |
| #1 | 753.0 / 766.4 | 1,465.0 / 1,477.7 | 4,053.9 / 4,079.2 | 62°C |
| #2 | 747.1 / 764.2 | 1,459.9 / 1,476.6 | 3,987.7 / 4,052.8 | 73°C |
| #3 | 747.2 / 769.7 | 1,457.9 / 1,479.9 | 3,963.7 / 4,004.4 | 72°C |
| #4 | 752.9 / 760.9 | 1,474.9 / 1,493.1 | 4,063.8 / 4,075.9 | 63°C |
| #5 | 761.3 / 776.3 | 1,478.7 / 1,498.6 | 4,086.4 / 4,095.7 | 61°C |
| #6 | 744.7 / 762.0 | 1,455.7 / 1,473.3 | 3,965.5 / 3,998.9 | 71°C |
| #7 | 730.5 / 747.7 | 1,424.6 / 1,445.9 | 3,936.0 / 3,978.0 | 73°C |
| **8-GPU mean** | **749 / 765** | **1,462 / 1,479** | **4,018 / 4,048** | — |

### B200 — node5701-c1

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 755.8 / 776.3 | 1,473.7 / 1,481.0 | 4,084.6 / 4,091.3 | 60°C |
| #1 | 755.7 / 775.2 | 1,478.1 / 1,489.8 | 4,051.6 / 4,071.5 | 62°C |
| #2 | 744.4 / 760.9 | 1,454.5 / 1,472.2 | 3,993.9 / 4,037.4 | 73°C |
| #3 | 739.0 / 767.5 | 1,443.4 / 1,455.8 | 3,951.6 / 3,968.1 | 72°C |
| #4 | 755.8 / 773.0 | 1,471.9 / 1,481.0 | 4,079.7 / 4,096.8 | 62°C |
| #5 | 761.4 / 766.4 | 1,481.0 / 1,494.2 | 4,065.3 / 4,091.3 | 61°C |
| #6 | 741.6 / 759.8 | 1,455.6 / 1,473.3 | 3,913.4 / 3,975.8 | 72°C |
| #7 | 741.7 / 762.0 | 1,435.8 / 1,456.9 | 3,970.9 / 4,008.8 | 73°C |
| **8-GPU mean** | **749 / 768** | **1,462 / 1,476** | **4,014 / 4,043** | — |

### B200 — node5702-c1

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 761.9 / 808.1 | 1,470.3 / 1,537.1 | 4,057.7 / 4,436.5 | 61°C |
| #1 | 757.2 / 803.7 | 1,460.8 / 1,527.2 | 4,017.5 / 4,398.0 | 62°C |
| #2 | 750.8 / 791.6 | 1,449.4 / 1,517.3 | 3,970.7 / 4,340.9 | 73°C |
| #3 | 753.1 / 800.4 | 1,455.8 / 1,529.4 | 3,914.8 / 4,278.2 | 73°C |
| #4 | 758.1 / 800.4 | 1,463.0 / 1,531.6 | 4,019.8 / 4,396.9 | 64°C |
| #5 | 760.3 / 804.8 | 1,473.1 / 1,541.5 | 4,036.5 / 4,415.6 | 62°C |
| #6 | 746.1 / 794.9 | 1,456.0 / 1,520.6 | 3,956.0 / 4,323.3 | 75°C |
| #7 | 731.9 / 787.3 | 1,444.3 / 1,522.8 | 3,875.0 / 4,208.9 | 75°C |
| **8-GPU mean** | **752 / 799** | **1,459 / 1,528** | **3,981 / 4,350** | — |

### B200 — node5800-c1

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 750.9 / 767.5 | 1,457.8 / 1,466.7 | 4,028.6 / 4,066.0 | 60°C |
| #1 | 757.9 / 776.3 | 1,478.7 / 1,490.9 | 4,071.9 / 4,090.2 | 63°C |
| #2 | 749.2 / 770.8 | 1,464.6 / 1,484.3 | 4,024.3 / 4,062.7 | 74°C |
| #3 | 737.4 / 763.1 | 1,444.1 / 1,452.5 | 3,982.5 / 4,020.9 | 72°C |
| #4 | 761.1 / 779.6 | 1,484.6 / 1,504.1 | 4,094.0 / 4,102.3 | 62°C |
| #5 | 758.5 / 778.5 | 1,484.7 / 1,493.1 | 4,088.2 / 4,100.1 | 61°C |
| #6 | 754.0 / 773.0 | 1,455.3 / 1,485.4 | 4,021.5 / 4,056.1 | 72°C |
| #7 | 744.6 / 769.7 | 1,447.2 / 1,471.1 | 3,913.7 / 3,948.3 | 72°C |
| **8-GPU mean** | **752 / 772** | **1,465 / 1,481** | **4,028 / 4,056** | — |

### B200 — node5801-c1

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 757.8 / 827.9 | 1,469.5 / 1,532.7 | 4,092.3 / 4,206.7 | 61°C |
| #1 | 753.0 / 822.4 | 1,459.8 / 1,522.8 | 4,050.5 / 4,160.6 | 61°C |
| #2 | 748.3 / 820.2 | 1,451.7 / 1,512.9 | 3,962.1 / 4,067.1 | 75°C |
| #3 | 751.0 / 820.2 | 1,457.5 / 1,516.2 | 4,037.7 / 4,135.3 | 71°C |
| #4 | 762.7 / 834.5 | 1,476.4 / 1,540.4 | 4,098.2 / 4,210.0 | 63°C |
| #5 | 760.6 / 836.7 | 1,466.6 / 1,528.3 | 4,087.2 / 4,196.8 | 60°C |
| #6 | 734.6 / 800.4 | 1,437.5 / 1,499.7 | 3,945.2 / 4,063.8 | 72°C |
| #7 | 741.7 / 810.3 | 1,440.2 / 1,499.7 | 3,986.2 / 4,092.4 | 73°C |
| **8-GPU mean** | **751 / 822** | **1,457 / 1,519** | **4,032 / 4,142** | — |

### B200 — node5802-c1

| GPU | FP32 (mean / max) | BF16 (mean / max) | FP8 (mean / max) | Peak temp |
|-----|-------------------|-------------------|------------------|-----------|
| #0 | 760.1 / 770.8 | 1,478.0 / 1,505.2 | 4,095.5 / 4,104.5 | 60°C |
| #1 | 741.7 / 756.5 | 1,452.3 / 1,464.5 | 4,005.5 / 4,063.8 | 63°C |
| #2 | 743.6 / 769.7 | 1,449.4 / 1,463.4 | 3,930.2 / 3,984.6 | 74°C |
| #3 | 746.4 / 765.3 | 1,460.0 / 1,478.8 | 4,000.5 / 4,050.6 | 71°C |
| #4 | 753.5 / 768.6 | 1,466.5 / 1,483.2 | 4,072.2 / 4,102.3 | 64°C |
| #5 | 763.0 / 776.3 | 1,485.6 / 1,492.0 | 4,091.6 / 4,104.5 | 60°C |
| #6 | 744.1 / 768.6 | 1,450.7 / 1,465.6 | 3,996.2 / 4,024.2 | 72°C |
| #7 | 739.5 / 764.2 | 1,442.8 / 1,453.6 | 3,972.2 / 3,996.7 | 72°C |
| **8-GPU mean** | **749 / 767** | **1,461 / 1,476** | **4,020 / 4,054** | — |

Across the 12 nodes, slots #2, #3, #6, #7 peak at 70–75°C, against 60–66°C for slots #0, #1, #4, #5, and slots #2, #6, #7 are the most often below their node's mean. The per-slot figures are in "Performance — Outliers and Variation" below.

## Performance — Outliers and Variation

Mean deviation of each B200 slot from its own node's mean, over 12 nodes (worst single node in parentheses):

| Slot | FP32 | BF16 | FP8 | Peak temp (mean, range) |
|---|---:|---:|---:|---:|
| #0 | +0.8% (-0.5%) | +0.8% (-1.1%) | +1.4% (+0.0%) | 60.6°C (60–61) |
| #1 | +0.7% (-1.0%) | +0.5% (-0.6%) | +0.9% (-0.4%) | 62.4°C (61–64) |
| #2 | -0.7% (-2.7%) | -0.6% (-2.6%) | -1.0% (-2.2%) | 73.6°C (72–75) |
| #3 | -0.3% (-1.9%) | -0.2% (-1.4%) | -0.7% (-1.7%) | 71.8°C (70–73) |
| #4 | +0.5% (-1.6%) | +0.4% (-1.9%) | +0.8% (-0.7%) | 63.8°C (62–66) |
| #5 | +0.9% (-0.4%) | +0.6% (-0.5%) | +1.1% (-1.3%) | 61.0°C (60–62) |
| #6 | -0.6% (-2.2%) | -0.4% (-1.4%) | -1.2% (-2.5%) | 72.3°C (71–75) |
| #7 | -1.2% (-2.7%) | -1.0% (-2.5%) | -1.3% (-2.8%) | 73.6°C (72–75) |

- **B200**: Slots #2, #6, #7 are the slowest and among the hottest in all three precisions, across all 12 nodes; slot #3 runs just as hot with a smaller throughput deficit. Spread within one node: FP32 1.9–4.4%, BF16 1.6–4.2%, FP8 2.5–4.6%. The largest single deviation from a node mean is -2.8%. No throttling, and the pattern follows the slot rather than the node, which points to chassis position (airflow, slot proximity).

## Expected Ceiling Comparison

| | B200 (per-GPU dense Tensor Core peak)* | Achieved | fraction of peak |
|---|---|---|---|
| TF32 (FP32 input) | ~1,125 TFLOPS | 749 | 0.67× |
| BF16 | ~2,250 TFLOPS | 1,457 | 0.65× |
| FP8 | ~4,500 TFLOPS | 4,010 | 0.89× |

*Dense, no sparsity. B200 sparse peaks are 2× these values.

The B200 FP8 path achieves 89% of dense peak, which is excellent. FP32/BF16 at 65–67% is normal for cuBLAS GEMM on this matrix size. Numbers are consistent with healthy B200 Tensor Cores.

## B200 vs. H200

| | FP32 | BF16 | FP8 |
|---|---|---|---|
| B200 (measured) | 749 TFLOPS | 1,457 TFLOPS | 4,010 TFLOPS |
| H200 (reference) | 368 TFLOPS | 713 TFLOPS | 1,468 TFLOPS |
| **B200 / H200** | **2.04×** | **2.04×** | **2.73×** |

The B200 delivers **~2.0× the throughput of an H200** in FP32 and BF16, consistent with Blackwell's generational leap over Hopper. The FP32:BF16 ratio is identical on both (1:1.94), confirming the same Tensor Core data-type scaling law. The gap widens sharply at FP8 — **2.7×** — because Blackwell's 5th-gen Tensor Cores have a substantially higher FP8 peak relative to Hopper's 4th-gen cores (B200 achieves 89% of dense peak vs H200's lower FP8 ceiling). For FP8-quantised inference workloads, the B200 advantage is disproportionately larger than the raw 2× marketing figure suggests.

## Thermals / Power

| | B200 peak temps (12 nodes) | Throttling |
|---|---|---|
| Coolest GPUs | #0, #5: 60–62°C | None |
| Mid | #1, #4: 61–66°C | None |
| Hottest | **#2, #3, #6, #7: 70–75°C** | None |

No throttling (HW, Thermal SW, or Thermal HW) on any GPU in any precision run.

## Stability

- No ECC errors, XID events, or NaN/inf reported in any of the 36 sub-runs.
- Each sub-run shows a short start-up transient, a long stable plateau (steady-state at large matrix), and a brief dip on the final measurement (teardown artifact, not real degradation).
- "All GPUs seem healthy" reported by gpu-fryer at the end of each of the 36 sub-runs.

---

## Verdict per GPU (all three precisions combined)

The B200 column covers that slot on all 12 nodes.

| GPU | B200 (12 nodes) |
|-----|-------------|
| #0 | **PASS** — +1.0% vs node mean (worst -1.1%), 61°C |
| #1 | **PASS** — +0.7% vs node mean (worst -1.0%), 62°C |
| #2 | **PASS** — -0.8% vs node mean (worst -2.7%), 74°C |
| #3 | **PASS** — -0.4% vs node mean (worst -1.9%), 72°C |
| #4 | **PASS** — +0.6% vs node mean (worst -1.9%), 64°C |
| #5 | **PASS** — +0.8% vs node mean (worst -1.3%), 61°C |
| #6 | **PASS** — -0.8% vs node mean (worst -2.5%), 72°C |
| #7 | **PASS** — -1.2% vs node mean (worst -2.8%), 74°C |

**Recommendation:** B200 slots #2, #3, #6, #7 run about 11°C hotter than the other four on every node, and #2, #6, #7 are consistently slightly slower across **every** precision. This looks like chassis position with reduced airflow rather than a fault (no throttling, within 2.8% of the node mean). Worth tracking: if the gap widens over time or any of these begins throttling, inspect the chassis airflow/cold-aisle for those slot positions.

---

## Side note: What is "dense peak"?

**Dense peak** is the vendor-published theoretical maximum Tensor-Core throughput for a given precision, with no sparsity. It is a hardware ceiling derived from silicon spec, not a measurement:

```
peak_FLOPS = (#Tensor Cores) × (ops per Tensor Core per cycle for that precision) × (boost clock)
```

"Ops per cycle" already encodes the precision — lower-precision formats pack more multiply-accumulates per cycle (on Blackwell, FP8 ≈ 2× BF16 ≈ 2× TF32). The published number uses the **boost clock**, so it is an upper bound rarely sustained in practice; the "sparse peak" sometimes quoted in marketing is 2× this, assuming 2:4 structured sparsity.
