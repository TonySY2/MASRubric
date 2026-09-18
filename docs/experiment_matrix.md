# Experiment Matrix

See the [main-method runbook](paper_main_reproduction.md) for model settings,
datasets, embedding caches, and grader setup.

Main-method experiments use `test/run_paper_main.py`:

```bash
python test/run_paper_main.py --list
python test/run_paper_main.py --help
python test/run_paper_main.py --suite math_8b --benchmark gsm8k --method adv2 --dry-run
```

Select `--suite math_8b`, `math_14b`, or `code_8b`, a `--benchmark` (or `all`),
and `--method adv2` or `baseline`. These runs use `paper/runtime/` and the
settings in `configs/paper_main.json`. Use `--preflight` to check required
resources and `--dry-run` to inspect commands.

After configuring the endpoints below, provide an embedding cache matching the
selected pool and embedding model:

```bash
# Math readiness check; no inference requests are sent.
python test/run_paper_main.py \
  --suite math_8b --benchmark gsm8k --method adv2 \
  --assets-root paper/assets \
  --embedding-cache-file /path/to/math_pool_embeddings.jsonl \
  --preflight

# Code smoke run: two records from the full input file.
python test/run_paper_main.py \
  --suite code_8b --benchmark mbpp --method adv2 \
  --assets-root paper/assets \
  --embedding-cache-file /path/to/code_pool_embeddings.jsonl \
  --limit 2
```

Use the full dataset as input; `--limit N` selects its first N records after
preflight checks.

The 8B suites use Qwen3-8B for reasoning and auditing, Qwen3.5-9B for selection,
and Qwen3-Embedding-8B for embeddings. To use other models, add
`--allow-model-override`. OlympiadBench supports a separate grader environment
via `--olympiad-python /path/to/legacy-env/bin/python`.

LiveCodeBench input is external. For one task, pass
`--benchmark livecode --in-file /path/to/historical_livecode_400.jsonl`; for
`--benchmark all`, place that historical 400-record file at
`paper/assets/datasets/livecode/livecodebench_v1.jsonl`. Use the original
400-record evaluation set.

| Experiment | Supported entry point |
| --- | --- |
| Dynamic-MAS + ADv2, 8B math and code | `run_paper_main.py`; suites `math_8b` / `code_8b`, method `adv2` |
| Dynamic-MAS + ADv2, 14B math | `run_paper_main.py --suite math_14b --method adv2` |
| Dynamic-MAS baseline | `run_paper_main.py --method baseline` with the selected suite |
| Historical Fixed-MAS, Single Agent, CoT, Self-Refine, PRM, Multi-TAG, ADv1 | Not included in the main-table launcher |
| Fixed framework and ablation variants | Development presets in `run_release_experiment.py`; see below |

The [result tables](release_results.md) retain the original reported scores;
new runs produce their own measurements. Release presets do not cover every
historical comparison or ablation implementation.

The sections below describe the release launcher. Its `--model-profile` value
only labels the output directory. It neither selects nor verifies the model;
the `*_MODEL` environment variables determine the actual models. Participant
temperatures and other sampling settings remain in the release agent code, and
the release launcher does not expose a seed option.

## Required Environment

Set OpenAI-compatible endpoints and model names with environment variables:

```bash
export SELECTOR_URL="http://host:port/v1"
export SELECTOR_MODEL="selector-model-name"
export REASONING_URL="http://host:port/v1"
export REASONING_MODEL="reasoning-model-name"
export SUPERVISOR_URL="http://host:port/v1"
export SUPERVISOR_MODEL="auditor-model-name"
export EMBEDDING_URL="http://host:port/v1"
export EMBEDDING_MODEL="embedding-model-name"

# Optional, defaults to EMPTY for local endpoints without auth.
export SELECTOR_KEY="EMPTY"
export REASONING_KEY="EMPTY"
export SUPERVISOR_KEY="EMPTY"
export EMBEDDING_KEY="EMPTY"
```

For optional pool overrides or externally hosted embedding caches, pass:

```bash
python test/run_release_experiment.py \
  --benchmark gsm8k --method adv2_math_main \
  --metric-pool-file /path/to/pool.json \
  --embedding-cache-file /path/to/pool_embeddings.jsonl
```

LiveCodeBench data is not bundled in this release. Provide a local jsonl file:

```bash
export AGENTDROPOUT_LIVECODE_FILE="/path/to/livecode.jsonl"
```

## Examples

List available benchmarks and method presets:

```bash
python test/run_release_experiment.py --list
```

Dry-run the main math configuration:

```bash
python test/run_release_experiment.py \
  --benchmark gsm8k \
  --method adv2_math_main \
  --model-profile math_8b \
  --dry-run
```

Dry-run mode prints endpoint placeholders when endpoint environment variables
are unset.

Run a small smoke subset:

```bash
python test/run_release_experiment.py \
  --benchmark gsm8k \
  --method adv2_math_main \
  --model-profile math_8b \
  --limit 2
```

Legacy `test/run-*.sh` scripts are thin wrappers over this launcher:

```bash
bash test/run-gsm8k.sh --method adv2_math_main --model-profile math_8b --limit 2
```

## Method Presets

| Preset | Release configuration | Main arguments |
| --- | --- | --- |
| `autogen_baseline` | Dynamic-MAS / AutoGen baseline | `--baseline_only` |
| `adv2_math_main` | Math ADv2 setting | `--retrieval_mode rerank --retrieve_p 20 --select_q 5 --batch_audit_metrics --pass_rate 0.6 --retries_times 3` |
| `adv2_math_iter2` | Table 4 iteration ablation | Same as main, `--retries_times 2` |
| `adv2_math_iter4` | Table 4 iteration ablation | Same as main, `--retries_times 4` |
| `adv2_math_top3` | Table 4 retrieved-indicator ablation | Same as main, `--select_q 3 --pass_rate 1.0` (pass 3/3) |
| `adv2_math_top7` | Table 4 retrieved-indicator ablation | Same as main, `--select_q 7` |
| `adv2_math_pass_2of5` | Table 4 pass-threshold ablation | Same as main, `--pass_rate 0.4` |
| `adv2_math_pass_5of5` | Table 4 pass-threshold ablation | Same as main, `--pass_rate 1.0` |
| `adv2_math_nondedup_pool` | Table 4 pool-deduplication ablation | Same as main, with the bundled non-deduplicated pool |
| `adv2_math_random_1to5` | Table 4 retrieval-control ablation | `--retrieval_mode random --random_k_min 1 --random_k_max 5` |
| `adv2_math_no_indicator_pool` | Table 4 no-pool control | Uses the low-level universal audit switch |
| `adv2_code_main` | Code ADv2 setting | `--retrieval_mode direct --direct_k 3 --batch_audit_metrics --pass_rate 1.0` |

The no-pool control activates the release universal audit path.

The release benchmark id `olymMATH` defaults to the Easy dataset. To run Hard,
explicitly pass `--in-file test/project_datasets/olymMATH/OlymMATH-EN-HARD.jsonl`.

## Indicator Pool Notes

The math and code indicator-pool JSON files are bundled in this
repository under `test/metrics_pool/`. Precomputed embedding caches can exceed
GitHub's single-file size limit, so they are optional release artifacts. Either
generate them locally or host them outside the repository and pass them to the
launcher with `--embedding-cache-file`.

For example, to generate the mixed code embedding cache:

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file test/metrics_pool/code_mixed/deduplicated_metrics_pool.json \
  --output_cache_file test/metrics_pool/code_mixed/deduplicated_embeddings-trigger.jsonl
```

For the math non-deduplication ablation:

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file test/metrics_pool/two_pool/mixed_metrics_two_pool.json \
  --output_cache_file test/metrics_pool/two_pool/mixed_embeddings_cache_two_pool.jsonl
```

Training-time scripts in `train/` collect raw trajectories for building a pool
and are controlled by environment variables. After collection, run:

```bash
cd train
python Extraction-deduplication-embedding.py
```

The extraction script produces deduplicated indicator records and embedding
caches that can be supplied to the test-time launcher.
