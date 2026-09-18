# Running main-method experiments

Use `test/run_paper_main.py` for Dynamic-MAS + ADv2 math and code experiments.
`configs/paper_main.json` lists the 13 tasks, model roles, final temperatures,
and historical reference results.

## Experiment settings

The runtime and assets come from `main_method_reference_20260711`.
Source-file provenance is listed in `paper/runtime_provenance.json`.

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
| Final temperature | 0.7 (`FinalRefer`) | 0.0 (`FinalWriteCodeMBPP` / `FinalWriteCode`) |
| Pool size | 2,000 | 2,545 |

The bundled code pool has no explicit `match_profile` entries. These indicators
use the supervisor's `any/any` defaults and apply across interfaces. Preflight
reports profile coverage when a different pool is supplied.

Each run manifest records the final class and its temperature. The selector uses
the service's sampling defaults; no seed is set. Use the same model weights and
serving/chat-template configuration when comparing runs.

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

## Supported scope

The archived source maps to Dynamic-MAS + ADv2 math 8B and code 8B.
`--method baseline` and `--suite math_14b` provide comparisons on this runtime.
Historical Fixed-MAS and other comparison methods are outside this entry point.
The July source archive includes later development, so historical scores in
[release_results.md](release_results.md) remain reference results rather than
a guarantee of identical new-run accuracy. Evaluate the complete datasets;
missing outputs count as incorrect.
