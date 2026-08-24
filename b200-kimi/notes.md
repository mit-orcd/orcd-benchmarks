# Can we apply MTP in our runs?

**On B200: no.** DSpark speculative decoding does not compose with pipeline parallelism
(vllm-project/vllm#50098), and PP2 is mandatory here because the 1561 GB Kimi-K3
checkpoint does not fit one 8×B200 node (1538 GB). Tested directly as the `lever2_spec`
arm in the improvement job — the recipe's own gate rejects the combination.

**On MI355X: yes, in principle.** MI355X fits the model on one node at TP8 (288 GB/GPU),
so no PP is needed and DSpark works — this is exactly what SemiAnalysis's
`kimik3_fp4_mi355x_mtp.sh` recipe does. But we have no MI355X hardware access; our
MI355X baseline is a pre-existing ATOM run from `../amd-benchmarks`, not something we
can rerun with MTP added.

Two further blockers even for a hypothetical B200 attempt: the speculator model
(`RedHatAI/Kimi-K3-speculator.dspark`) is not downloaded here, and the container runs
`HF_HUB_OFFLINE=1`.

**Bottom line:** not on the hardware we have, in the configuration it forces.

See `results/kimi-k3-base-b200.md` §7.4 for the measured MTP effect (SemiAnalysis's own
data, ~2.7× per-user speed at c=1) and why it is unavailable to B200 specifically.
