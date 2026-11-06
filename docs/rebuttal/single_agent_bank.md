# Single-agent control with the same bank

`scripts/rebuttal/single_bank.py` implements a single-agent bank control for MATH-500 and AQuA. It instantiates the recovered `MathSolverMath500` or `MathSolverAqua` and the original `supervisor_reasoning_pick_metric.Supervisor` directly.

The protocol is one solver answer episode: one initial answer and up to three feedback-driven revisions. It uses the reference Math Solver role, prompts and temperatures; the same math bank and cached embeddings; Top-20 retrieval followed by the original supervisor's reranker selecting up to five criteria; batched auditing with the original per-criterion fallback; and the 0.6 acceptance threshold. Criteria are retrieved again on each revision, as in the recovered math preset.

Only the group-chat turn selector is removed. The criterion reranker remains inside the Supervisor. The default role mapping is Qwen3-8B reasoning and supervision, Qwen3-Embedding-8B embeddings, and Qwen3.5-9B as the documented but unused group-chat selector. There is no second final-decision model and no whole-team fallback in this single-agent protocol.

| Method | Audit and revision | Failed terminal answer |
|---|---|---|
| `vanilla` | None | Publish the initial answer |
| `full` | Same bank, up to three revisions | Abstain |
| `revise-only` | Same bank, up to three revisions | Publish the last answer |

An abstention is an intentional unsolved outcome, counted as incorrect in fixed-denominator accuracy. It is distinct from an unknown outcome caused by an API failure or timeout. The terminal candidate is preserved and graded separately even when withheld. The reference empty-judgment policy remains pass, with its source recorded by the mechanism trace.

```bash
# Set REASONING_URL, SUPERVISOR_URL and EMBEDDING_URL, plus their optional
# *_MODEL and *_KEY variables. Model names default to the reference mapping.
python scripts/rebuttal/single_bank.py --benchmark math500 --method full \
  --ids 0,1 --output-dir runs/single-bank-math500-full

# The default command displays a plan. Add --execute to make model requests.
python scripts/rebuttal/single_bank.py --benchmark math500 --method full \
  --ids 0,1 --output-dir runs/single-bank-math500-full --execute
```

Use fresh directories for `vanilla`, `full` and `revise-only`; use `--benchmark aqua` for AQuA and omit `--ids` for the full dataset. Dataset, bank and cache paths can be provided explicitly. Execution validates the bank schema, criterion names, finite vectors and complete cache coverage before constructing the model clients. `--allow-model-override` is for explicitly named model substitutions and local fixtures; every mismatch is recorded.

Each sample directory contains `api_calls.jsonl` with requests, responses, returned usage and failed logical calls; the reference `usage.jsonl`; `mechanism.jsonl` with each audit and action; and a result record. Prompts contain the problem and, for AQuA, its choices. Gold answers and rationales are used only by the local grader after inference. An API/usage failure stops subsequent samples, preserving completed results and missing IDs in the receipt. SDK-internal retries cannot provide token usage for failed server attempts; such calls make cost accounting incomplete.

This runner measures realized cost; it does not enforce a total-token cap. `sampling-budget.json` contains each successfully accounted episode's total LLM prompt-plus-completion tokens, including retrieval summaries, reranking, auditing and revisions. Embedding tokens are recorded separately. The budget file is directly compatible with the sampling control:

```bash
python scripts/rebuttal/sampling_vote.py run --benchmark math500 --ids 0,1 \
  --budget-file runs/single-bank-math500-full/sampling-budget.json \
  --output-dir runs/single-bank-math500-vote --execute
```

Incomplete usage never becomes a zero-cost budget. Compare measured single-agent bank outcomes with Dynamic-MAS on the same problem IDs, and report abstentions, unknown outcomes and cost together. Fixed-denominator accuracy and final candidate accuracy answer different questions; the latter does not count withheld answers as delivered answers.
