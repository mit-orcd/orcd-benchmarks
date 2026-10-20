# Kimi-K3 on 8 × MI355X: ATOM + DSpark MTP

Run: `/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/node6100/kimi-atom-mtp/atom/kimi_atom_mtp_20261016_194820`. Image `rocm/atom-dev:nightly_202610011450`, our ATOM baseline flags (TP8, max-num-seqs 64, no prefix caching), ISL/OSL 1024/1024, `--ignore-eos`, 10 × users prompts; `--method dspark --draft-model Inferact/Kimi-K3-DSpark`, real acceptance (no forced acceptance length). TPOT = time per output token, TTFT = time to first token. No MTP = the no-MTP arm of this run (same image).

| Users | draft length | out tok/s | per-user tok/s | TPOT med (ms) | TTFT med (ms) | per-user, no MTP | MTP / no MTP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 197.4 | 256.8 | 3.89 | 174 | 64.0 | 4.01x |
| 2 | 2 | 280.7 | 161.5 | 6.19 | 189 | – | – |
| 4 | 2 | 464.1 | 138.5 | 7.22 | 192 | – | – |
| 8 | 2 | 811.5 | 112.2 | 8.91 | 196 | 56.4 | 1.99x |
| 10 | 2 | 695.5 | 78.4 | 12.76 | 215 | – | – |
| 16 | 2 | 1,117.0 | 78.2 | 12.78 | 212 | – | – |
| 32 | 2 | 1,810.2 | 61.7 | 16.20 | 218 | 35.3 | 1.75x |
| 48 | 2 | 2,149.0 | 49.2 | 20.34 | 231 | – | – |
| 64 | 2 | 2,463.1 | 41.7 | 23.96 | 239 | – | – |
