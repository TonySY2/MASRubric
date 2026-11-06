# Mechanism controls and audit traces

The `review_mechanism` family uses the recovered Dynamic-MAS runtime for all five conditions. Configure the same model roles, dataset, bank and embedding cache for each comparison.

| Condition | Revision budget | Failed terminal message | Purpose |
|---|---:|---|---|
| `baseline` | No audit | Broadcast | Original unaudited comparison |
| `masrubric` | 3 | Withhold from downstream participant and final-decision inputs | Full method |
| `revise_only` | 3 | Broadcast | Independent effect of withholding |
| `withhold_only` | 0 | Withhold | Effect without revision |
| `audit_only` | 0 | Broadcast | Auditing with neither revision nor withholding |

`revise_only` changes only `--withholding-policy broadcast`. The retrieval, audit prompts, threshold, revision prompt and revision budget stay the same. Successful audits and empty audits already pass in both conditions. Each method can develop a different downstream trajectory after its first terminal failure; equal revision limits do not imply equal realized token consumption.

The preserved communication mechanism filters messages when a participant constructs its input and when the final decision receives its history. The group-chat selector still sees the group conversation. If the full method retains at most one message, the historical runtime clears the scoreboard and reruns the task without auditing. This fallback remains active; its incidence and tokens are part of the method comparison.

## Prepare and execute

Create a new plan path and output directory for each run. Planning and command checking make no model requests. The initial small run checks operational behavior; the full comparison uses all benchmark examples and at least three fresh repetitions.

```bash
python scripts/rebuttal/run_matrix.py plan \
  --family review_mechanism --suite math_8b --benchmarks gsm8k,aime24,math500 \
  --repeats 1 --limit 10 --plan plans/mechanism-pilot.json \
  --output-dir runs/mechanism-pilot
python scripts/rebuttal/run_matrix.py check --plan plans/mechanism-pilot.json

# After configuring the model endpoints:
python scripts/rebuttal/run_matrix.py run --plan plans/mechanism-pilot.json

# Prepare the full math comparison separately; configure code roles for code_8b.
python scripts/rebuttal/run_matrix.py plan \
  --family review_mechanism --suite math_8b --repeats 3 \
  --plan plans/mechanism-math.json --output-dir runs/mechanism-math
```

The same launcher accepts individual controlled runs:

```bash
python test/run_paper_main.py --suite math_8b --benchmark gsm8k \
  --variant-name revise_only --withholding-policy broadcast \
  --output-dir runs/gsm8k-revise-only
python test/run_paper_main.py --suite code_8b --benchmark mbpp \
  --variant-name withhold_only --retry-budget 0 \
  --output-dir runs/mbpp-withhold-only
```

The comparison table should show paired problem outcomes, macro and problem-weighted accuracy, repetitions, total tokens and fallback frequency. Use the existing `summarize.py` tools for final task outcomes. A single-agent same-bank control and a [token-budget sampling curve](sampling_controls.md) complement the mechanism comparison.

## Empty audits: exact runtime policy

`Supervisor.judge` passes a message when its returned judgment list is empty. The satisfaction rate is recorded as `null` in that case, with `empty_policy: "pass"`; the implementation does not divide by zero or assign an observed rate of 100%.

There are separate cases:

1. The reranker returns no usable criterion names. A valid empty selection, names outside the candidate set, or an empty filtered candidate set can produce zero selected criteria. A rerank API/parse exception instead falls back to candidate Top-Q.
2. Criteria were selected, but individual audits failed through all five attempts. The returned judgments can be empty or partially missing; the historical decision uses the judgments that were returned. Missing judgments are recorded separately from the no-selected-criteria case.
3. The optional cheap precheck can bypass the audit. The main launcher fixes this option to off.
4. A baseline or fallback run disables auditing entirely. Such messages are not zero-criterion audit attempts.

`mechanism.jsonl` records selected names/count, selection mode and fallback, returned judgments/count, exhausted audit failures, empty cause, revision feedback, threshold decision and final communication action. Each final score also contains `audit_attempts`. Append-only `message_finalized` events preserve the first phase when a fallback subsequently clears its scoreboard. Final score files from older runs cannot establish which empty-audit case occurred or count all revised attempts. Reconstructing the reported 14B 8.7% requires its original per-attempt logs and the original denominator.

New tracing preserves the historical default decisions. It does not retrospectively label historical empty audits, nor establish the accuracy effect of changing their policy. Report selection-empty and audit-failure-empty rates separately when interpreting efficiency.

## Double review of revisions and false positives

Export a uniform sample of message episodes with every generated version and its available input context:

```bash
python scripts/rebuttal/audit_traces.py runs/example/gsm8k/result.json \
  --mechanism-trace runs/example/gsm8k/mechanism.jsonl \
  --sample 100 --seed 20261106 --output-dir annotations/example
```

Give `annotation_reviewer_a.json` and `annotation_reviewer_b.json` to different reviewers. Keep `annotation_key.private.json` separate. The review forms omit model audit verdicts, feedback and withholding outcomes. Each reviewer enters their identity and labels every version `correct`, `incorrect` or `unverifiable`, with a reason and an error location. A partial contribution can be correct without solving the whole problem. Use `unverifiable` when the supplied problem/context cannot establish correctness; request reference solutions independently if needed. Labels start empty.

```bash
python scripts/rebuttal/audit_traces.py runs/example/gsm8k/result.json \
  --annotation-key annotations/example/annotation_key.private.json \
  --labels-a annotations/example/annotation_reviewer_a.json \
  --labels-b annotations/example/annotation_reviewer_b.json \
  --output-dir annotations/example-reviewed
```

The report counts adjacent revision transitions and initial-to-final transitions separately. Correct→incorrect measures spoiling; incorrect→correct measures correction. Its false-positive numerator is failed audits of versions both humans label correct, divided by all human-agreed correct audited versions in the sample. It also reports the distinct count of correct versions with any flawed criterion, since one failed criterion need not fail the threshold. Pending, unverifiable and disagreement counts remain visible; disagreement requires adjudication. Final task correctness is never substituted for an intermediate-message label. These message transitions establish local effects; the final-answer effect requires the paired task runs above.

For older result files without attempt histories, the exporter supplies only the final message. Its labels can support final-message audit checks, but cannot recover an absent before/after revision pair.
