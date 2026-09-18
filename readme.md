# AgentDropoutV2

This repository anonymously releases code, data, and reproducibility materials
for **AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via
Test-Time Rectify-or-Reject Pruning**.

<p align="center">
  <img src="image/readme/AgentDropoutV2-logo.png" alt="AgentDropoutV2 Logo" width="200">
</p>

## News

- **2026-05-27**: arXiv preprint released:
  [arXiv:2602.23258](https://arxiv.org/abs/2602.23258).
- **2026-05-22**: Math and code indicator-pool JSON files are bundled in the
  GitHub release.
- **2026-02-27**: Initial code and dataset release.

## Overview

AgentDropoutV2 is a test-time framework for improving information flow in
multi-agent systems without retraining the base agents. During MAS execution it:

1. intercepts each agent output before broadcast,
2. retrieves failure-driven indicators from an offline pool,
3. audits the output and provides targeted rectification feedback,
4. rejects unreleased outputs that still fail the audit threshold,
5. falls back to the original MAS path when pruning would collapse the team.

<p align="center">
  <img src="image/readme/adv1-vs-adv2.png" alt="ADv1 versus ADv2 overview">
</p>

<p align="center">
  <img src="image/readme/main-picture.png" alt="AgentDropoutV2 framework">
</p>

## Repository Layout

```text
test/run_paper_main.py             Historical main-method reproduction entry point.
configs/release_experiments.json   Release benchmarks, pools, and method presets.
docs/experiment_matrix.md          Main-table coverage and release configuration guide.
docs/release_results.md            Historical Table 1 / Table 2 / Table 3 / Table 4 snapshot.
test/                             Test-time inference, benchmark loaders, and public launcher.
train/                            Training-time collection and indicator-pool construction.
image/readme/                     README figures.
```

## Requirements

Use Python 3.10. The full historical environment is pinned in
`requirements.txt`; for a fresh setup:

```bash
conda create -n agentdropoutv2 python=3.10
conda activate agentdropoutv2
pip install -r requirements.txt
```

The runners use OpenAI-compatible chat and embedding endpoints. Local vLLM
servers can use `EMPTY` keys when authentication is disabled.

## Main-table reproduction

Start with the dedicated historical main-method entry point:

```bash
python test/run_paper_main.py --list
python test/run_paper_main.py --help
python test/run_paper_main.py --suite math_8b --benchmark gsm8k --method adv2 --dry-run
```

This entry point uses a separate runtime recovered from the frozen experiments;
it does not forward to the release presets. Its supported runs follow the
recovered experiment settings. See
[the experiment matrix](docs/experiment_matrix.md) for coverage and limitations.
The [main-method runbook](docs/paper_main_reproduction.md) covers the exact
model roles, external assets, two grader environments, and source boundaries.
The result tables are historical records, not scores measured by a fresh run of
this release. Matching a method name or a few retrieval flags does not establish
equivalence: agent prompts, audit rules, model settings, input subsets, and
evaluation denominators also matter.

After setting the endpoint variables below, check the bundled historical assets
and a matching external embedding cache:

```bash
python test/run_paper_main.py \
  --suite math_8b --benchmark gsm8k --method adv2 \
  --assets-root paper/assets \
  --embedding-cache-file /path/to/math_pool_embeddings.jsonl \
  --preflight
```

For a two-question smoke run, replace `--preflight` with `--limit 2`. The
preflight reads the full input file even for a smoke run, so pass the full
archived dataset and let `--limit` select records. Code runs require the code
pool's matching cache, not the math cache.

The release launcher below remains available for development and new experiments.
Its fixed framework is a reconstruction using the release agents; it is not an
equivalent runner for the historical Table 1 Fixed-MAS rows.

## Release Quick Start

Set endpoint variables:

```bash
export SELECTOR_URL="http://host:port/v1"
export SELECTOR_MODEL="selector-model-name"
export REASONING_URL="http://host:port/v1"
export REASONING_MODEL="reasoning-model-name"
export SUPERVISOR_URL="http://host:port/v1"
export SUPERVISOR_MODEL="auditor-model-name"
export EMBEDDING_URL="http://host:port/v1"
export EMBEDDING_MODEL="embedding-model-name"

export SELECTOR_KEY="EMPTY"
export REASONING_KEY="EMPTY"
export SUPERVISOR_KEY="EMPTY"
export EMBEDDING_KEY="EMPTY"
```

List available benchmarks and method presets:

```bash
python test/run_release_experiment.py --list
```

Preview the command shape without configured endpoints:

```bash
python test/run_release_experiment.py \
  --benchmark gsm8k \
  --method adv2_math_main \
  --model-profile math_8b \
  --limit 2 \
  --dry-run
```

Run the release math configuration on a small smoke subset:

```bash
python test/run_release_experiment.py \
  --benchmark gsm8k \
  --method adv2_math_main \
  --model-profile math_8b \
  --limit 2
```

The old per-benchmark shell scripts are now thin wrappers over the same launcher:

```bash
bash test/run-gsm8k.sh --method adv2_math_main --model-profile math_8b --limit 2
```

In the release launcher, `--model-profile` is an output-directory label only: it
does not select, load, or validate a model. All actual served model names come
from the endpoint environment. Use `--method adv2_code_main --model-profile
code_8b` for the release code configuration. Use the historical entry point above
when reproducing a supported main-table experiment.

## Fixed framework and token usage

The default framework remains `dynamic`. Add `--framework fixed` to use the
reconstructed FullGraph-style schedule: five agents, all forward DAG edges, and one
complete round. `--fixed-rounds N` controls complete fixed rounds; `--max-turns`
controls dynamic chat only. Fixed runs reuse the release agents/prompts, pass
only accepted predecessor outputs downstream, and aggregate the last round.
They do not use a selector or the dynamic framework's whole-task fallback.
These scheduling properties have offline tests; historical prompt, audit, data,
and score equivalence has not been established for this reconstructed runner.

```bash
# Fixed MAS baseline, then Fixed MAS + ADv2 (use adv2_code_main for code).
python test/run_release_experiment.py --benchmark gsm8k --method fixed_baseline --limit 2
python test/run_release_experiment.py --benchmark gsm8k --method adv2_math_main --framework fixed --limit 2
```

Token accounting is automatic for both frameworks. Results contain per-question
`token_usage`; the printed `*.usage.summary.json` contains run totals, alongside
per-call `*.usage.jsonl` and per-sample `*.samples.jsonl`. Each launch has its own
output directory under `test/results_release/<model-profile>/<framework>/`.
Counts include selector, all reasoning/rectification attempts, summary, rerank,
audit, final decision, and dynamic fallback. LLM and embedding tokens are separate.
Missing provider usage stays `null` with `complete: false`; `observed_*` fields
show known partial sums. These are response-usage measurements; transport retries
for which the provider supplies no usage cannot be reconstructed.

## Indicator Pools

The math and code indicator-pool JSON files are bundled in this repository:

```text
test/metrics_pool/two_pool/deduped-mixed_metrics_two_pool.json
test/metrics_pool/two_pool/mixed_metrics_two_pool.json
test/metrics_pool/code_mixed/deduplicated_metrics_pool.json
```

Precomputed embedding caches can exceed GitHub's single-file size limit. They
are optional release artifacts: either generate them locally, or host them
outside the repository and pass them at runtime. For optional local overrides,
pass explicit launcher arguments:

```bash
python test/run_release_experiment.py \
  --benchmark gsm8k --method adv2_math_main \
  --metric-pool-file /path/to/pool.json \
  --embedding-cache-file /path/to/pool_embeddings.jsonl
```

For example, to generate the mixed code embedding cache:

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file test/metrics_pool/code_mixed/deduplicated_metrics_pool.json \
  --output_cache_file test/metrics_pool/code_mixed/deduplicated_embeddings-trigger.jsonl
```

For the math non-deduplication ablation, generate the matching cache with:

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file test/metrics_pool/two_pool/mixed_metrics_two_pool.json \
  --output_cache_file test/metrics_pool/two_pool/mixed_embeddings_cache_two_pool.jsonl
```

To build a custom pool:

```bash
cd train
bash run-math-train.sh
python Extraction-deduplication-embedding.py
```

The training scripts are controlled by environment variables and write their
outputs to the configured local paths.

## Common Arguments

| Argument | Description |
| --- | --- |
| `--in_file` / `--out_file` | Input dataset and output result path. |
| `--log_file` | Detailed run log path. |
| `--selector_url`, `--selector_model`, `--selector_key` | Selector/planner endpoint. |
| `--reasoning_url`, `--reasoning_model`, `--reasoning_key` | Participant and final-answer endpoint. |
| `--supervisor_url`, `--supervisor_model`, `--supervisor_key` | Auditor endpoint. |
| `--embedding_url`, `--embedding_model`, `--embedding_key` | Embedding endpoint for retrieval. |
| `--metric_pool_file`, `--embedding_cache_file` | Indicator pool and precomputed embedding cache. |
| `--baseline_only` | Run the MAS baseline without audit/pruning. |
| `--retrieval_mode` | `direct`, `rerank`, or `random`. |
| `--retrieve_p`, `--select_q` | Rerank path: retrieve top-P candidates, then select up to Q indicators. |
| `--direct_k` | Direct retrieval top-K. |
| `--random_k_min`, `--random_k_max` | Random indicator-count range for retrieval-control ablations. |
| `--batch_audit_metrics` | Audit all selected indicators in one batched auditor call. |
| `--pass_rate` | Fraction of selected indicators that must pass. |
| `--retries_times` | Rectification retry budget for one agent output. |
| `--limit` | Optional subset size for smoke tests. |

## Privacy

The public release should not contain private endpoint URLs, API keys, personal
paths, server IPs, or local runtime settings. The launcher reads secrets from
environment variables and masks key values when it prints commands.

## Citation

```bibtex
@misc{wang2026agentdropoutv2optimizinginformationflow,
      title={AgentDropoutV2: Optimizing Information Flow in Multi-Agent Systems via Test-Time Rectify-or-Reject Pruning},
      author={Yutong Wang and Siyuan Xiong and Xuebo Liu and Wenkang Zhou and Liang Ding and Miao Zhang and Min Zhang},
      year={2026},
      eprint={2602.23258},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.23258},
}
```

## Acknowledgments

This codebase builds on [AgentDropout](https://github.com/wangzx1219/AgentDropout).
