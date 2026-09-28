<p align="center">
  <img src="image/readme/masrubric-logo.png" alt="MASRubric network logo" width="220">
</p>

<h1 align="center">MASRubric</h1>

<p align="center">
  <strong>Auditing Information Flow in Multi-Agent Systems<br>
  with Failure-Distilled Pitfall Rubrics</strong>
</p>

<p align="center">
  <a href="https://github.com/TonySY2/MASRubric/actions/workflows/tests.yml"><img src="https://github.com/TonySY2/MASRubric/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
</p>

<p align="center">
  <a href="#paper-and-citation">Manuscript</a> &nbsp;|&nbsp;
  <a href="docs/paper_main_reproduction.md">Runbook</a> &nbsp;|&nbsp;
  <a href="docs/experiment_matrix.md">Experiments</a> &nbsp;|&nbsp;
  <a href="#reported-results">Reported results</a>
</p>

## Overview

MASRubric audits intermediate messages before errors propagate through a
multi-agent system. It distills recurring reasoning pitfalls from failed
trajectories into a reusable criterion bank, then retrieves a context-specific
rubric for each message. The auditor's verdict controls whether that message is
passed downstream, revised by its author, or withheld.

The framework addresses three requirements for useful test-time rubrics:

- **Test-time ready:** reusable criteria apply to unseen inputs without a
  reference answer or human-written criterion at inference time.
- **Situation-specific:** each message is checked against criteria whose
  applicability conditions match its reasoning context.
- **Actionable:** criterion-level judgments lead to **Pass**, **Revise**, or
  **Withhold** decisions before the message reaches downstream agents.

<p align="center">
  <img src="image/readme/masrubric-overview.png" alt="Existing rubrics versus MASRubric on test-time readiness, situation specificity, and actionable decisions" width="1000">
</p>

<p align="center"><em>Figure 1 from the MASRubric manuscript: existing rubrics and MASRubric compared against the three requirements.</em></p>

## Method

1. **Mine criteria from failures.** Run a source MAS on training queries, retain
   failed trajectories, and identify recurring agent-level pitfalls. Each
   criterion records a name, diagnostic definition, applicability condition,
   and inspection directive.
2. **Compact the bank.** Retrieve semantically similar criteria and use an LLM
   to eliminate duplicate error patterns. The resulting bank stays fixed
   during inference.
3. **Retrieve a contextual rubric.** Match the message's reasoning situation to
   applicability conditions, then select the relevant criteria.
4. **Audit and intervene.** Aggregate binary criterion judgments into a
   satisfaction rate. Pass messages meeting the threshold; otherwise return
   diagnostic feedback to the original agent for revision, or withhold the
   message when its revision budget is exhausted.

Reference answers support offline failure collection and rubric mining. The
online auditor operates without them.

<p align="center">
  <img src="image/readme/masrubric-framework.png" alt="Offline criterion-bank construction and online MASRubric information flow" width="1100">
</p>

<p align="center"><em>Figure 2 from the MASRubric manuscript. Lower: offline rubric mining and two-stage deduplication. Upper: contextual retrieval, criterion-level auditing, and Pass / Revise / Withhold decisions.</em></p>

## Reported results

The current MASRubric manuscript reports the following average accuracies with
Qwen3-8B reasoning agents (Tables 1 and 2):

| Evaluation | Underlying MAS | + MASRubric | Gain |
| --- | ---: | ---: | ---: |
| Fixed-MAS, 9 math benchmarks | 51.15% | 52.74% | +1.59 points |
| Dynamic-MAS, 9 math benchmarks | 50.86% | 53.69% | +2.83 points |
| Dynamic-MAS, 4 code benchmarks | 46.63% | 48.37% | +1.74 points |

These are manuscript-reported results. New runs produce their own measurements;
the release's supported reproduction scope is described in the
[runbook](docs/paper_main_reproduction.md#supported-scope-and-runtime-behavior).

## Quick start

The commands below use **Python 3.10 and Bash on Linux or WSL2**. Model serving is
configured separately through OpenAI-compatible endpoints.

```bash
git clone https://github.com/TonySY2/MASRubric.git
cd MASRubric
python3.10 -m venv .venv-paper
source .venv-paper/bin/activate
python -m pip install -r paper/requirements.txt

# Inspect supported tasks and a command without making model calls.
python test/run_paper_main.py --list
python test/run_paper_main.py --suite math_8b --benchmark gsm8k --method masrubric --dry-run
```

Configure the four model roles. Replace the example hosts with your own services
and set each `*_KEY` to its key. `EMPTY` is only for a local service without
authentication.

```bash
export REASONING_URL=http://reasoning-host:8000/v1 REASONING_MODEL=Qwen3-8B
export SUPERVISOR_URL=http://auditor-host:8000/v1 SUPERVISOR_MODEL=Qwen3-8B
export SELECTOR_URL=http://selector-host:8000/v1 SELECTOR_MODEL=Qwen3.5-9B
export EMBEDDING_URL=http://embedding-host:8000/v1 EMBEDDING_MODEL=Qwen3-Embedding-8B
export REASONING_KEY=EMPTY SUPERVISOR_KEY=EMPTY SELECTOR_KEY=EMPTY EMBEDDING_KEY=EMPTY
```

Generate a cache for the bundled math criterion bank. This step calls the
embedding service; keep the same embedding model for subsequent experiments.

```bash
python test/metrics_pool/two_pool/embed_metrics-trigger.py \
  --input_file paper/assets/pools/math/deduped-mixed_metrics_two_pool.json \
  --output_cache_file paper/assets/pools/math/deduped-mixed_two_pool-trigger.jsonl

# Check local assets and dependencies, then run two questions.
python test/run_paper_main.py --suite math_8b --benchmark gsm8k --method masrubric --preflight
python test/run_paper_main.py --suite math_8b --benchmark gsm8k --method masrubric --limit 2
```

Remove `--limit` for the full benchmark. Use `--method baseline` for the baseline
on the same runtime. Alternative model roles require `--allow-model-override`.
See the [runbook](docs/paper_main_reproduction.md) for code tasks, the separate
OlympiadBench grader, external assets, and output interpretation.

## Repository contents

| Path | Purpose |
| --- | --- |
| `test/run_paper_main.py`, `configs/paper_main.json` | Reference Dynamic-MAS launcher and benchmark settings |
| `paper/runtime/`, `paper/assets/` | Reference runtime, evaluation data, and criterion banks |
| `test/run_release_experiment.py`, `configs/release_experiments.json` | Configurable dynamic/fixed experiments and intervention variants |
| `test/masrubric/` | Configurable runtime and per-call token accounting |
| `train/` | Offline failure collection and criterion-bank construction |
| `tests/` | Launcher, runtime, failure-handling, and accounting tests |

The reference launcher covers nine math tasks and four code tasks. The banks
contain 2,000 math criteria and 2,545 code criteria. Twelve evaluation datasets
are bundled; LiveCodeBench's matching 400-problem file and embedding caches must
be supplied or generated separately. Raw experiment outputs are excluded; new
runs write their own measurements.

## Additional experiments

```bash
python test/run_release_experiment.py --list
python test/run_release_experiment.py --benchmark gsm8k --method masrubric_math_main \
  --framework fixed --model-profile math_8b --limit 2 --dry-run
```

The [experiment matrix](docs/experiment_matrix.md) describes available presets,
environment variables, and custom banks. Its `--model-profile` option labels
outputs; endpoint variables select the actual models.

## Validation and scope

This release contains a recovered reference implementation with later
maintenance. Exact historical source identity and reproduction of every paper
result are not established. The configurable fixed DAG is an extension; it does
not establish equivalence with the paper's Fixed-MAS rows. Detailed behavior,
sampling differences, and supported comparisons are recorded in the
[runbook](docs/paper_main_reproduction.md#supported-scope-and-runtime-behavior).

Run the offline and local-fixture checks in the installed environment:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Tests use local fixtures and do not establish model quality or benchmark scores.
Code benchmarks execute generated Python; use an isolated environment. Generated
results, logs, caches, and local credentials are ignored by Git. Review any
manually added outputs before sharing them, as they can contain prompts, paths,
and endpoint configuration.

## Paper and citation

This repository presents **MASRubric: Auditing Information Flow in Multi-Agent
Systems with Failure-Distilled Pitfall Rubrics**. Its public preprint link and
citation will be added when the manuscript update is available.

This codebase builds on [AgentDropout](https://github.com/wangzx1219/AgentDropout).
