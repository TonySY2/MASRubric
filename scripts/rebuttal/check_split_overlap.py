#!/usr/bin/env python3
"""Compare exact normalized problem text across explicit train/test exports."""
import argparse
import json
from pathlib import Path
import re
import unicodedata

from prepare_assets import iter_records

TEXT_FIELDS = ("question_content", "question", "problem", "prompt", "text", "task_prompt")
ID_FIELDS = ("id", "task_id", "problem_id", "question_id", "unique_id", "name")


def collect(paths):
    questions = {}
    count, missing = 0, 0
    for path in paths:
        for index, row in enumerate(iter_records(path)):
            count += 1
            value = next((row.get(k) for k in TEXT_FIELDS if isinstance(row.get(k), str) and row[k].strip()), None)
            if value is None:
                missing += 1
                continue
            text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()
            identity = str(next((row[k] for k in ID_FIELDS if k in row), index))
            questions.setdefault(text, []).append({"file": str(path), "id": identity, "row": index})
    return questions, {"records": count, "missing_problem_text": missing, "unique_normalized_questions": len(questions)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, action="append", required=True)
    p.add_argument("--test", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train, train_summary = collect(args.train)
    test, test_summary = collect(args.test)
    shared = sorted(set(train) & set(test))
    report = {"comparison": "exact problem text after Unicode NFKC and whitespace normalization",
              "train": train_summary, "test": test_summary,
              "text_coverage_complete": train_summary["records"] > 0 and test_summary["records"] > 0 and train_summary["missing_problem_text"] == 0 and test_summary["missing_problem_text"] == 0,
              "overlap_question_count": len(shared),
              "matches": [{"train": train[t], "test": test[t]} for t in shared],
              "coverage": "Provided exports only; this comparison does not detect paraphrases or model pretraining exposure."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "matches"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
