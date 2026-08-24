# SemiAnalysis InferenceX — reference configs for Kimi-K3

Fetched 2026-08-24 from the Apache-2.0 open-source repo behind the
<https://inferencex.semianalysis.com> dashboard:

    https://github.com/SemiAnalysisAI/InferenceX   (formerly InferenceMAX)

| File | Upstream path |
|---|---|
| `agg-b200-tp8pp2-agentic.yaml` | `benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k3/agentic/` |
| `kimik3_fp4_mi355x_mtp.sh` | `benchmarks/single_node/agentic/` |
| `kimik3_fp4_mi355x_atom_mtp.sh` | `benchmarks/single_node/agentic/` |

**These are configs, not results.** The dashboard's measured numbers are served from a
backend the public repo does not contain, so no numeric comparison against our sweep is
possible from here. What these files DO establish is the exact methodology, which is what
section 7 of `../results/kimi-k3-base-b200.md` compares against ours.
