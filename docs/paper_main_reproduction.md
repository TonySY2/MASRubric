# Running the recovered main method

Use `test/run_paper_main.py` for the recovered Dynamic-MAS + ADv2 execution
chain. The separate `test/run_release_experiment.py` remains the portable release
implementation and fixed-framework extension. A matching method name or a few
matching numeric arguments does not establish that these implementations are
equivalent.

## What was recovered

The source is `main_method_reference_20260711`: its source workspace, per-task
configurations and summaries, indicator pools, datasets, and the separate code
grader. `paper/runtime_provenance.json` lists the selected source files and the
portability changes. `configs/paper_main.json` maps the 13 tasks to their source
runs, full denominators, correct counts, and actual model-role names.

The reference package itself states that its workspace had development changes
after the April/May source runs. This release recovers that documented reference;
it cannot certify that the July snapshot is byte-for-byte the code executed in
April/May. No full model benchmark has been rerun as part of this repair.

| Setting | Math main | Code main |
| --- | --- | --- |
| Source run | 2026-04-23 math9 main | 2026-05-08 code4 main |
| Framework | Dynamic SelectorGroupChat | Dynamic SelectorGroupChat |
| Participant/supervisor | Qwen3-8B | Qwen3-8B |
| Selector | Qwen3.5-9B | Qwen3.5-9B |
| Embedding | Qwen3-Embedding-8B | Qwen3-Embedding-8B |
| Team prompts | Actual verifier variant | Original code roles and MBPP training examples |
| Retrieval | top 20, select 1–5 | direct top 3, profile-aware query path |
| Audit | Batch, pass fraction 0.6 | Batch, pass fraction 1.0 |
| Turns / rectification | 7 / 3 | 7 / 3 |
| Pool size | 2,000 | 2,545 |

The code reference pool has **zero explicit `match_profile` entries**. The
preserved supervisor normalizes absent profiles to `any/any`. Although its
profile-aware query and filtering code are enabled, this does not establish
effective interface filtering for those generic indicators. The launcher reports
coverage and a warning; it does not invent profiles or replace the original pool.

Sampling parameters come from the preserved source rather than the reference
README's blanket prose. For example, the recovered code final-decision classes
use temperature 0.0, whereas the reference README says 0.7. The selector does
not explicitly set temperature. No source seed or fully pinned serving/chat
template configuration was recorded. Service defaults remain a reproducibility
boundary.

## Install the two grader environments

Use Python 3.10. The original OlympiadBench grader depends on an older ANTLR
runtime that conflicts with the OlymMATH verifier. Keep the interpreters separate:

```bash
python3.10 -m venv .venv-paper
.venv-paper/bin/pip install -r paper/requirements.txt
python3.10 -m venv .venv-olympiad
.venv-olympiad/bin/pip install -r paper/requirements-olympiad.txt
```

`--olympiad-python .venv-olympiad/bin/python` selects the latter interpreter only
for OlympiadBench. Preflight imports and exercises the actual graders; missing
dependencies cannot silently switch the run to string comparison.

## Assets and endpoints

Twelve evaluation datasets, the MBPP training examples, and the two indicator
pools are included under `paper/assets/`. LiveCodeBench is external because the
preserved 400-record file is approximately 1.27 GB. Place it at
`paper/assets/datasets/livecode/livecodebench_v1.jsonl`, or use `--in-file` for a
single LiveCodeBench run. Arbitrary newer LiveCodeBench exports are not this
historical 400-problem benchmark.

Embedding caches are external. Supply the original matching cache using
`--embedding-cache-file`, or place the math and code files at the paths listed in
`configs/paper_main.json`. An `--assets-root` with the same `datasets/` and
`pools/` layout can reuse an existing reference package. Preflight checks dataset
counts, indicator names, and embedding dimensions. It does not certify the
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

The launcher checks model names against the historical role mapping, accepting
directory or organization prefixes. Served names alone do not prove the actual
weights or chat template. `--allow-model-override` explicitly permits an
intentional variant or a fake test endpoint and records the mismatch. Keys are
passed to the clients and excluded from saved manifests.

## Commands

```bash
# Inspect all 13 benchmark mappings without model calls.
python test/run_paper_main.py --list
python test/run_paper_main.py --suite math_8b --benchmark gsm8k --dry-run

# Validate files, role configuration, and grader availability; no model requests.
.venv-paper/bin/python test/run_paper_main.py --suite math_8b --preflight \
  --olympiad-python .venv-olympiad/bin/python \
  --embedding-cache-file /path/to/math-cache.jsonl

# One-sample wiring check, followed by all nine complete math tasks.
.venv-paper/bin/python test/run_paper_main.py --suite math_8b --benchmark gsm8k \
  --limit 1 --embedding-cache-file /path/to/math-cache.jsonl
.venv-paper/bin/python test/run_paper_main.py --suite math_8b \
  --olympiad-python .venv-olympiad/bin/python \
  --embedding-cache-file /path/to/math-cache.jsonl

# Four complete code tasks; the external LiveCodeBench file must be prepared.
.venv-paper/bin/python test/run_paper_main.py --suite code_8b \
  --embedding-cache-file /path/to/code-cache.jsonl
```

`--benchmark olymmath_easy` and `--benchmark olymmath_hard` are separate tasks.
`--limit N` selects the first N examples and labels results as `smoke_subset`.
The default concurrency is one; change it deliberately with `--concurrency`.
Each launch uses a fresh directory, so old answers cannot be silently resumed.
Preflight is local validation; a real smoke run is needed to check model services.

Each task writes a redacted `manifest.json`, raw results, `usage.jsonl`, process
and detailed logs, and a `summary.json`. Code outputs are rejudged by the original
`paper_code_eval.py`, without trusting stored correctness flags. Summary accuracy
uses the selected/full benchmark denominator; missing results count as incorrect
and also make completion fail. A failed child, duplicate result IDs, missing
outputs, or grader failure returns a nonzero launcher exit status. Partial suites
do not receive a completed-suite macro score. Historical usage logs retain their
original schema; they do not inherit the release runtime's complete/observed
token-accounting contract.

## Scope of the result claims

The recovered source records support Dynamic-MAS + ADv2 math 8B and code 8B.
`--method baseline` is a comparison on that same recovered runtime, and
`--suite math_14b` is a model-transfer entrypoint. Neither is certified here as
the exact source of its separately published historical row. The original
FullGraph/V1 rows and other comparison methods need their own source/configuration
mapping; the release `--framework fixed` extension must not be relabeled as their
historical reproduction.

The corrected published-table column assignments and historical denominator
notes are in [release_results.md](release_results.md). Source-run missing outputs
were counted as incorrect, including 5/165 CodeContests problems; rerunning should
attempt the complete dataset rather than reproduce the historical omissions.
