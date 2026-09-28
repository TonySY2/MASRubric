# Running MASRubric Experiments

Use `test/run_paper_main.py` for Dynamic-MAS + MASRubric math and code experiments.
`configs/paper_main.json` lists the 13 tasks, model roles, final temperatures,
and reference model-role requirements. Precomputed experimental results are
excluded; each new run records its own measurements.

## Install the two grader environments

Use Python 3.10 with Bash on Linux or WSL2. The commands below are run from the
repository root. Coding executors and graders use Unix signal-based timeouts;
native Windows execution is not supported by these instructions.

The original OlympiadBench grader depends on an older ANTLR
runtime that conflicts with the OlymMATH verifier. Keep the interpreters separate:

```bash
python3.10 -m venv .venv-paper
.venv-paper/bin/pip install -r paper/requirements.txt
python3.10 -m venv .venv-olympiad
.venv-olympiad/bin/pip install -r paper/requirements-olympiad.txt
```

`paper/requirements.txt` installs client/runtime and grader dependencies; model
servers run separately. The root `requirements.txt` is the broader historical
development environment, including GPU-serving dependencies, and is not needed
for this quick start.

`--olympiad-python .venv-olympiad/bin/python` selects the latter interpreter only
for OlympiadBench. Preflight imports and exercises the actual graders; missing
dependencies cannot silently switch the run to string comparison.

## Assets and endpoints

Twelve evaluation datasets, the MBPP training examples, and the two criterion
banks are included under `paper/assets/`. The banks contain 2,000 math and 2,545
code criteria, matching the current manuscript's Table 5. The 400-record
LiveCodeBenchV1 input is external. Place it at
`paper/assets/datasets/livecode/livecodebench_v1.jsonl`, or use `--in-file` for a
single LiveCodeBench run. Use the matching 400-problem evaluation set when
comparing with the paper; a different export changes the evaluation population.

Embedding caches are external. Supply a matching cache using
`--embedding-cache-file`, or place the math and code files at the paths listed in
`configs/paper_main.json`. An `--assets-root` with the same `datasets/` and
`pools/` layout can reuse an existing reference package. Preflight checks dataset
counts, criterion names, and embedding dimensions. It does not certify the
semantic identity of a regenerated embedding cache.

Configure the existing OpenAI-compatible services:

```bash
export REASONING_URL=http://reasoning-host:8000/v1
export REASONING_MODEL=Qwen3-8B
export SUPERVISOR_URL=http://supervisor-host:8000/v1
export SUPERVISOR_MODEL=Qwen3-8B
export SELECTOR_URL=http://selector-host:8000/v1
export SELECTOR_MODEL=Qwen3.5-9B
export EMBEDDING_URL=http://embedding-host:8000/v1
export EMBEDDING_MODEL=Qwen3-Embedding-8B
export REASONING_KEY=EMPTY SUPERVISOR_KEY=EMPTY SELECTOR_KEY=EMPTY EMBEDDING_KEY=EMPTY
```

The launcher checks model names against the reference role mapping, accepting
directory or organization prefixes. Served names alone do not prove the actual
weights or chat template. `--allow-model-override` explicitly permits an
intentional variant or a fake test endpoint and records the mismatch. Keys are
passed to the clients and excluded from saved manifests.

### Generate matching embedding caches

With `EMBEDDING_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_KEY` configured, generate
the caches at the reference launcher's default paths:

```bash
.venv-paper/bin/python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file paper/assets/pools/math/deduped-mixed_metrics_two_pool.json \
  --output_cache_file paper/assets/pools/math/deduped-mixed_two_pool-trigger.jsonl
.venv-paper/bin/python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file paper/assets/pools/code/deduplicated_metrics_pool.json \
  --output_cache_file paper/assets/pools/code/deduplicated_embeddings-trigger.jsonl
```

These commands send requests to the embedding service. Use the same model and
criterion bank during inference. The generator resumes by criterion name; use a
fresh output file after changing the model or bank, then pass that path with
`--embedding-cache-file`. Preflight checks completeness and dimensions before a
run. The examples below show explicit external cache paths; omit that option
when using the default paths above.

## Commands

```bash
# Inspect all 13 benchmark mappings without model calls.
python test/run_paper_main.py --list
python test/run_paper_main.py --suite math_8b --benchmark gsm8k --method masrubric --dry-run

# Validate files, role configuration, and grader availability; no model requests.
.venv-paper/bin/python test/run_paper_main.py --suite math_8b --method masrubric --preflight \
  --olympiad-python .venv-olympiad/bin/python \
  --embedding-cache-file /path/to/math-cache.jsonl

# One-sample wiring check, followed by all nine complete math tasks.
.venv-paper/bin/python test/run_paper_main.py --suite math_8b --benchmark gsm8k --method masrubric \
  --limit 1 --embedding-cache-file /path/to/math-cache.jsonl
.venv-paper/bin/python test/run_paper_main.py --suite math_8b --method masrubric \
  --olympiad-python .venv-olympiad/bin/python \
  --embedding-cache-file /path/to/math-cache.jsonl

# Four complete code tasks; the external LiveCodeBench file must be prepared.
.venv-paper/bin/python test/run_paper_main.py --suite code_8b --method masrubric \
  --embedding-cache-file /path/to/code-cache.jsonl
```

`--benchmark olymmath_easy` and `--benchmark olymmath_hard` are separate tasks.
`--limit N` selects the first N examples and labels results as `smoke_subset`.
The default concurrency is one; change it deliberately with `--concurrency`.
Each launch uses a fresh directory, so old answers cannot be silently resumed.
Preflight is local validation; a real smoke run is needed to check model services.

Each task writes a redacted `manifest.json`, raw results, `usage.jsonl`, process
and detailed logs, and a `summary.json`. Code outputs are rejudged by the bundled
`paper_code_eval.py`, without trusting stored correctness flags. Summary accuracy
uses the selected/full benchmark denominator; missing results count as incorrect
and also make completion fail. A failed child, duplicate result IDs, missing
outputs, or grader failure returns a nonzero launcher exit status. Partial suites
do not receive a completed-suite macro score. Reference runtime usage logs retain
their own schema; they do not inherit the release runtime's complete/observed
token-accounting contract.

## Supported scope and runtime behavior

The recovered implementation supports Dynamic-MAS + MASRubric math 8B and code 8B.
`--method baseline` and `--suite math_14b` provide comparisons on this runtime.
The paper's Fixed-MAS and other comparison methods are outside this entry point.
The reported tables describe the paper's experiments; this recovered runtime
does not guarantee identical new-run accuracy. Evaluate the complete datasets;
missing outputs count as incorrect.

The configurable runtime in `test/masrubric/` also provides a fixed DAG and
intervention presets through `test/run_release_experiment.py`. This extension
has not been established as equivalent to the Fixed-MAS experiments reported
in the current manuscript or every comparison and ablation method. See the
[experiment matrix](experiment_matrix.md) for its available presets.

The current manuscript reports six chat turns (§4.1). This artifact sets its
message limit to 7, counting the initial user task plus at most 6 agent messages.
The manuscript also specifies a three-revision budget, 20 retrieval candidates,
up to 5 selected criteria for math and 3 for code, and satisfaction thresholds
of 0.6 and 1.0, respectively. The artifact matches those rubric-size limits,
revision budgets, and thresholds. Satisfaction is measured over the selected
criteria; the math threshold is 3/5 only when five criteria are selected.

Two code-path differences remain: direct top-3 retrieval replaces the
coarse-to-fine selection described in §3.3, and the final-answer temperature is
0.0 while §4.1 states 0.7 for non-auditor models. Math final-answer temperature
is 0.7. These implementation differences should be considered when comparing
new runs with manuscript results.

Batch audits use the diagnostic definition, applicability condition, and
inspection directive. The manuscript's judgment stage uses only the
applicability condition and inspection directive (§3.3; Figure 7), as do the
artifact's single-criterion audits and per-criterion fallback. The preserved
runtime accepts an empty judgment set; API or parsing failures can yield empty or
partial judgments. Inspect these separately from successful criterion checks.
The dynamic runtime restarts a task without intervention when at most one usable
message remains; the fixed runtime does not use this whole-task fallback.

The code bank has no populated `match_profile` entries. The reference runtime
normalizes missing profiles to `any`; enabling its profile-aware path alone
does not establish effective hard filtering. Further provenance is recorded in
[`paper/runtime_provenance.json`](../paper/runtime_provenance.json).

## Criterion-bank terminology

Stored fields and command-line options retain their implementation names:

| Method term | Field or option |
| --- | --- |
| Criterion identifier | `name` |
| Diagnostic definition | `detailed_definition` |
| Applicability condition | `trigger_condition` |
| Inspection directive | `risk_alert` |
| Criterion bank | `--metric-pool-file`, `metrics_pool/` |
| Selected rubric size | `--select_q` or `--direct_k` |
| Rubric satisfaction threshold | `--pass_rate` |
| Revision budget | `--retries_times` |

One bank entry is a criterion; the selected set is a rubric. Scripts in `train/`
collect failed trajectories and construct custom banks without updating the
reasoning model's weights.

## Validation and output handling

Run the checks with the installed runtime dependencies:

```bash
.venv-paper/bin/python -m unittest discover -s tests -p 'test_*.py'
```

The suite tests launch contracts, actual SDK/agent wiring against a local HTTP
fixture, failure handling, and token accounting. It does not call external model
services or establish model quality. Code benchmarks execute generated Python;
run them in an isolated environment.

The configurable runtime records per-call and per-sample usage, separating LLM
and embedding tokens. Missing provider usage remains unknown with
`complete: false`. Reference-runtime logs retain their own schema. Generated
results, logs, caches, and local environment files are ignored by Git; review
any manually added output for prompts, paths, and endpoint configuration before
sharing it.
