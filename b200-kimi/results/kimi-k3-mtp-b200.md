# Kimi-K3 on 2 × 8 B200 with DSpark MTP (job 26092372)

TP8 × PP2, `vllm/vllm-openai:kimi-k3`, draft `Inferact/Kimi-K3-DSpark` (with the `pard_token` shim), DCP 1, ISL/OSL 1024/1024, `--ignore-eos`, max-num-seqs 64, 10 × conc prompts. TPOT = time per output token, TTFT = time to first token. Per-user tok/s = 1000 / median TPOT. No-spec = job 20916742.

## mtp7: draft length 7, attention backend TOKENSPEED_MLA

Server did not start: `(APIServer pid=3219376) NotImplementedError: Pipeline parallelism is not supported for this model. Supported models implement the `SupportsPP` interface.`

## mtp7d: draft length 7, attention backend default

Server did not start: `(APIServer pid=3248389) NotImplementedError: Pipeline parallelism is not supported for this model. Supported models implement the `SupportsPP` interface.`

