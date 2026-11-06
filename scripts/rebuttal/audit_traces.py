#!/usr/bin/env python3
"""Summarize final message audit records and export a blinded diagnostic sample."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from summarize import read_records, sample_id


def audit_records(result_file, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else Counter()
    seen = set()
    for row in read_records(result_file):
        identity = sample_id(row)
        if identity in seen:
            raise ValueError(f"Duplicate sample id {identity}")
        seen.add(identity)
        diagnostics["source_samples"] += 1
        scores = row.get("scores")
        if scores is None or scores == {}:
            diagnostics["samples_without_score_records"] += 1
            continue
        if not isinstance(scores, dict):
            diagnostics["malformed_score_containers"] += 1
            continue
        for message_id, score in scores.items():
            if not isinstance(score, dict):
                diagnostics["malformed_score_records"] += 1
                continue
            message = score.get("message") or {}
            content = message.get("content", "") if isinstance(message, dict) else str(message)
            judgments = score.get("judgements")
            state = "missing" if judgments is None else ("available" if isinstance(judgments, list) and all(isinstance(j, dict) for j in judgments) else "malformed")
            if state != "available":
                diagnostics[state + "_judgment_messages"] += 1
                judgments = None
            yield {"sample_id": identity, "message_id": str(message_id),
                   "question": row.get("question") or row.get("task_prompt") or row.get("problem") or "",
                   "message": content, "role": score.get("role"),
                   "judgements": judgments, "judgements_state": state,
                   "withheld": score.get("is_pruned") if isinstance(score.get("is_pruned"), bool) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261106, help="Diagnostic sampling seed only")
    args = p.parse_args()
    if args.sample < 0:
        p.error("--sample cannot be negative")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    diagnostics = Counter()
    rows = list(audit_records(args.result, diagnostics))
    size = Counter(len(row["judgements"]) for row in rows if row["judgements_state"] == "available")
    withheld = Counter("unknown" if row["withheld"] is None else str(row["withheld"]).lower() for row in rows)
    report = {"source_result": str(args.result), "final_scored_messages": len(rows),
              "input_diagnostics": dict(diagnostics),
              "empty_final_judgment_messages": size.get(0, 0),
              "final_rubric_size_histogram": dict(sorted(size.items())),
              "withholding": dict(withheld),
              "denominator": "Final message score records, not all audit attempts or LLM calls",
              "empty_cause": "Unknown from final scores alone; inspect retrieval, parse and API traces."}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selected = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
    labels, key = [], []
    for i, row in enumerate(selected):
        label_id = f"audit_{i + 1:04d}"
        labels.append({"annotation_id": label_id, "question": row["question"],
                       "message": row["message"], "role": row["role"],
                       "contains_material_error": None, "error_location": "",
                       "requires_additional_context": None, "independent_reason": "", "annotator": ""})
        key.append({"annotation_id": label_id, **row})
    for name, value in (("annotation_blind.json", labels), ("annotation_key.private.json", key)):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
