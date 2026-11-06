#!/usr/bin/env python3
"""Summarize run receipts and compare paired sample outcomes without hiding gaps."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import statistics

CODE_BENCHMARKS = {"mbpp", "humaneval", "codecontest", "livecode"}
GROUP_FIELDS = ("suite", "runtime", "family", "benchmark", "method", "expected", "full_benchmark_total")


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Non-object result row in {path}")
                    yield row
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            for row in value:
                if not isinstance(row, dict):
                    raise ValueError(f"Non-object result row in {path}")
                yield row
        elif isinstance(value, dict):
            for key, row in value.items():
                if not isinstance(row, dict):
                    raise ValueError(f"Expected sample mapping in {path}")
                yield {"id": key, **row}
        else:
            raise ValueError(f"Unsupported result object: {path}")


def sample_id(row):
    for key in ("id", "task_id", "problem_id", "question_id", "instance_id"):
        if key in row:
            return str(row[key])
    raise ValueError("Result row has no sample identifier")


def outcome(row, benchmark=None):
    # A runtime timeout or transport failure remains unresolved even when an
    # old harness serialized its default success flag as false.
    if row.get("status") in ("failed", "timeout", "interrupted", "no_result") or row.get("timed_out"):
        return None
    if row.get("error") in ("sample_timeout", "api_error", "generation_failed") or row.get("exception"):
        return None
    if "_paper_grader_outcome" in row:
        return row["_paper_grader_outcome"]
    if benchmark == "aqua" and not any(k in row for k in ("is_correct", "is_solved")):
        gold, pred = row.get("answer"), row.get("hypothesis")
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        if str(gold or "").strip().upper() not in set("ABCDE"):
            return None
        return str(gold).strip().upper() == str(pred or "").strip().upper()
    for key in ("is_correct", "is_solved", "correct", "passed"):
        if key in row:
            value = row[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            return None
    return None


def normalized_id(identity, benchmark=None):
    identity = str(identity)
    if benchmark == "codecontest":
        # The preserved runner appends an existing 8-character suffix to numeric
        # ids. Match its original row id without generating a new checksum.
        match = re.fullmatch(r"([0-9]+)_[0-9a-fA-F]{8}", identity)
        if match:
            return match.group(1)
    return identity


def indexed(path, benchmark=None):
    result = {}
    for row in read_records(path):
        identity = normalized_id(sample_id(row), benchmark)
        if identity in result:
            raise ValueError(f"Duplicate sample id {identity} in {path}")
        result[identity] = row
    grader = Path(path).parent / "paper_grader.json"
    if grader.exists():
        report = json.loads(grader.read_text(encoding="utf-8"))
        if report.get("trust_stored_flags") is not False:
            raise ValueError(f"Grader did not independently execute tests: {grader}")
        if report.get("total_records") != len(result):
            raise ValueError(f"Grader result count disagrees with results: {grader}")
        unmatched = {normalized_id(i, benchmark) for i in report.get("missing_dataset_records", [])}
        if not unmatched <= result.keys() or report.get("matched_records") != len(result) - len(unmatched):
            raise ValueError(f"Grader matching counts disagree: {grader}")
        changes = {}
        for mismatch in report.get("mismatches", []):
            identity = normalized_id(mismatch["id"], benchmark)
            if identity not in result or identity in unmatched or identity in changes or not isinstance(mismatch["rejudged"], bool):
                raise ValueError(f"Invalid grader mismatch: {grader}")
            changes[identity] = mismatch["rejudged"]
        for identity, row in result.items():
            # The sidecar lists changed verdicts; an unchanged known stored
            # verdict was also re-executed. Unmatched/unstored cases stay unknown.
            verdict = None if identity in unmatched else changes.get(identity, outcome(row, benchmark))
            result[identity] = dict(row, _paper_grader_outcome=verdict)
        verdicts = [row["_paper_grader_outcome"] for row in result.values()]
        lower = sum(v is True for v in verdicts)
        upper = lower + sum(v is None for v in verdicts) - len(unmatched)
        if not lower <= report.get("correct", -1) <= upper:
            raise ValueError(f"Grader correct count disagrees with per-sample verdicts: {grader}")
    return result


def expected_ids(job, candidates):
    """Prefer the runtime's frozen manifest; otherwise inspect the selected input."""
    manifests = []
    for path in candidates:
        manifest = path.parent / "manifest.json"
        if manifest.is_file():
            value = json.loads(manifest.read_text(encoding="utf-8"))
            if "sample_ids" in value:
                manifests.append(value["sample_ids"])
    if manifests:
        if len(manifests) != len(candidates):
            raise ValueError("Some result files lack the expected sample-id manifest")
        values = [normalized_id(i, job["benchmark"]) for ids in manifests for i in ids]
        source = "runtime_manifest"
    elif job.get("dataset_file") and Path(job["dataset_file"]).is_file():
        path = Path(job["dataset_file"])
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                raw = [json.loads(line) for line in stream if line.strip()]
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                nested = next((raw[key] for key in ("data", "test", "problems", "questions") if isinstance(raw.get(key), list)), None)
                if nested is not None:
                    raw = nested
                elif all(isinstance(row, dict) for row in raw.values()):
                    raw = list(raw.values())
        if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
            return None, "unavailable_input_layout"
        keys = ("id", "task_id", "problem_id", "question_id", "unique_id")
        if job["benchmark"] == "humaneval":
            keys = ("task_id", "name")
        elif job["benchmark"] == "mbpp":
            keys = ("task_id",)
        elif job["benchmark"] == "livecode":
            keys = ("question_id", "id")
        elif job["benchmark"].startswith("olymmath"):
            keys = ("unique_id", "id")
        elif job["benchmark"] == "codecontest":
            keys = ("name", "id") if job.get("runtime") == "release" else ("id", "name")
        values = [normalized_id(next((row[k] for k in keys if k in row), i), job["benchmark"])
                  for i, row in enumerate(raw[:job["expected"]])]
        source = "selected_input_ids"
    else:
        return None, "unavailable"
    if len(values) != job["expected"] or len(set(values)) != len(values):
        raise ValueError("Expected sample identities are duplicated or disagree with the planned denominator")
    return set(values), source


def summarize_job(job):
    root = Path(job["output_dir"])
    candidates = sorted(set(root.rglob("result.json")) | set(root.rglob("result.jsonl")) |
                        set(root.rglob(f"{job['benchmark']}_{job['method']}.json")))
    rows = {}
    for path in candidates:
        for identity, row in indexed(path, job["benchmark"]).items():
            if identity in rows:
                raise ValueError(f"Duplicate sample {identity} across result files in {root}")
            rows[identity] = row
    expected = job["expected"]
    if len(rows) > expected:
        raise ValueError(f"{root}: {len(rows)} rows exceed planned denominator {expected}")
    identities, identity_source = expected_ids(job, candidates)
    unexpected = sorted(set(rows) - identities) if identities is not None else []
    missing_ids = sorted(identities - set(rows)) if identities is not None else []
    values = [outcome(row, job["benchmark"]) for identity, row in rows.items() if identity not in unexpected]
    correct = sum(value is True for value in values)
    incorrect = sum(value is False for value in values)
    unknown = sum(value is None for value in values)
    missing = len(missing_ids) if identities is not None else expected - len(rows)
    requires_grader = job.get("runtime") == "paper" and job["benchmark"] in CODE_BENCHMARKS
    grader_ok = not requires_grader or bool(candidates) and all((p.parent / "paper_grader.json").is_file() for p in candidates)
    # The reference recorder drops calls with absent usage, so its observed
    # total is not evidence that accounting is complete.
    usage_files = sorted(root.rglob("*.usage.summary.json"))
    usage = [json.loads(path.read_text(encoding="utf-8")) for path in usage_files]
    usage_complete = bool(usage) and all(
        u.get("token_usage", {}).get(bucket, {}).get("complete") is True
        for u in usage for bucket in ("llm", "embedding"))
    observed_llm = observed_embedding = 0
    if usage:
        for item in usage:
            token_usage = item.get("token_usage", {})
            observed_llm += token_usage.get("llm", {}).get("observed_total_tokens", 0) or 0
            observed_embedding += token_usage.get("embedding", {}).get("observed_total_tokens", 0) or 0
    else:
        for path in root.rglob("usage.jsonl"):
            for event in read_records(path):
                if event.get("kind") == "embedding":
                    observed_embedding += event.get("total_tokens", 0) or 0
                else:
                    observed_llm += event.get("total_tokens", 0) or 0
    return {"expected": expected, "recorded": len(rows), "correct": correct,
            "incorrect": incorrect, "unknown": unknown, "missing": missing,
            "unexpected": len(unexpected), "unexpected_ids": unexpected, "missing_ids": missing_ids,
            "identity_validation": identity_source if identities is None else ("passed:" + identity_source if not unexpected and not missing_ids else "failed:" + identity_source),
            "grader_validation": "passed" if requires_grader and grader_ok else ("missing" if requires_grader else "runtime_verdicts"),
            "complete": missing == 0 and unknown == 0 and not unexpected and grader_ok,
            "accuracy_full_denominator": correct / expected if expected else None,
            "accuracy_scored_only": correct / (correct + incorrect) if correct + incorrect else None,
            "usage_complete": usage_complete if usage else None,
            "observed_llm_tokens": observed_llm, "observed_embedding_tokens": observed_embedding,
            "llm_total_tokens": observed_llm if usage_complete else None,
            "result_files": [str(p) for p in candidates]}


def paired(left, right, iterations=10000, seed=20261106, benchmark=None):
    a, b = indexed(left, benchmark), indexed(right, benchmark)
    shared = sorted(set(a) & set(b))
    known = [(outcome(a[k], benchmark), outcome(b[k], benchmark)) for k in shared]
    pairs = [(int(x), int(y)) for x, y in known if x is not None and y is not None]
    if not pairs:
        raise ValueError("No matched samples with known outcomes")
    diffs = [y - x for x, y in pairs]
    random_state = random.Random(seed)
    draws = sorted(sum(random_state.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(iterations))
    win = sum(x == 0 and y == 1 for x, y in pairs)
    loss = sum(x == 1 and y == 0 for x, y in pairs)
    n = win + loss
    # Integer division computes a bounded ratio without converting 2**n or
    # binomial sums to an overflowing float on larger benchmarks.
    pvalue = min(1.0, (2 * sum(math.comb(n, k) for k in range(min(win, loss) + 1))) / 2**n) if n else 1.0
    return {"left_records": len(a), "right_records": len(b), "shared_ids": len(shared),
            "left_only": len(set(a) - set(b)), "right_only": len(set(b) - set(a)),
            "paired_scored": len(pairs), "paired_unknown": len(shared) - len(pairs),
            "right_minus_left_accuracy": statistics.mean(diffs),
            "paired_bootstrap_95ci": [draws[int(.025 * (iterations - 1))], draws[int(.975 * (iterations - 1))]],
            "right_wins": win, "left_wins": loss, "mcnemar_exact_two_sided_p": pvalue,
            "bootstrap_iterations": iterations, "bootstrap_seed": seed,
            "scope": "matched scored examples; report missing and unknown alongside this comparison"}


def summarize_groups(rows):
    groups = {}
    for row in rows:
        key = tuple(row.get(k) for k in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in groups.items():
        complete = [r["accuracy_full_denominator"] for r in members if r["complete"] and r["status"] == "completed"]
        output.append({**dict(zip(GROUP_FIELDS, key)), "runs_total": len(members), "runs_complete": len(complete),
                       "accuracy_mean": statistics.mean(complete) if complete else None,
                       "accuracy_sample_sd": statistics.stdev(complete) if len(complete) > 1 else None,
                       "repeat_kind": "fresh_stochastic_run_no_seed_contract"})
    return output


def summarize_macros(rows, config=None):
    """A paper macro requires one complete full run of every suite benchmark."""
    if config is None:
        path = Path(__file__).resolve().parents[2] / "configs/paper_main.json"
        config = json.loads(path.read_text(encoding="utf-8"))
    keys = ("suite", "runtime", "family", "method", "repeat")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    output = []
    for key, members in groups.items():
        suite = key[0]
        domain = "code" if suite == "code_8b" else "math" if suite in ("math_8b", "math_14b") else None
        expected = {name: spec["total"] for name, spec in config["benchmarks"].items() if spec["domain"] == domain}
        by_benchmark = {}
        for row in members:
            by_benchmark.setdefault(row["benchmark"], []).append(row)
        missing = sorted(set(expected) - set(by_benchmark))
        unexpected = sorted(set(by_benchmark) - set(expected))
        duplicates = sorted(name for name, items in by_benchmark.items() if len(items) != 1)
        subset, invalid_denominator, incomplete = [], [], []
        for name, items in by_benchmark.items():
            if name not in expected:
                continue
            for row in items:
                if row.get("expected") != expected[name]:
                    subset.append(name)
                if row.get("full_benchmark_total") != expected[name]:
                    invalid_denominator.append(name)
                accuracy = row.get("accuracy_full_denominator")
                if not row.get("complete") or row.get("status") != "completed" or not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 1:
                    incomplete.append(name)
        complete = bool(expected) and not any((missing, unexpected, duplicates, subset, invalid_denominator, incomplete))
        macro = statistics.mean(by_benchmark[name][0]["accuracy_full_denominator"] for name in expected) if complete else None
        output.append({**dict(zip(keys, key)), "complete": complete,
                       "expected_benchmarks": sorted(expected), "recorded_benchmarks": sorted(by_benchmark),
                       "missing_benchmarks": missing, "unexpected_benchmarks": unexpected,
                       "duplicate_benchmarks": duplicates, "subset_benchmarks": sorted(set(subset)),
                       "invalid_full_denominator_benchmarks": sorted(set(invalid_denominator)),
                       "incomplete_benchmarks": sorted(set(incomplete)),
                       "macro_accuracy": macro, "macro_accuracy_percent": 100 * macro if macro is not None else None,
                       "scope": "one complete fixed full math-9 or code-4 suite per repeat; no per-benchmark run selection"})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, nargs="?")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--paired", type=Path, nargs=2, metavar=("CONTROL", "TREATMENT"))
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--benchmark", help="Use aqua for legacy AQuA option-only results")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.iterations < 100:
        p.error("--iterations must be at least 100")
    if args.paired:
        report = paired(*args.paired, iterations=args.iterations, benchmark=args.benchmark)
    else:
        if args.root is None:
            p.error("root is required unless --paired is used")
        report = {"jobs": [], "groups": []}
        for path in sorted(args.root.rglob("job.json")):
            job = json.loads(path.read_text(encoding="utf-8"))
            summary = summarize_job(job)
            row = {k: job.get(k) for k in ("id", "repeat", "status", "elapsed_seconds", *GROUP_FIELDS)}
            row.update(summary)
            report["jobs"].append(row)
        report["groups"] = summarize_groups(report["jobs"])
        report["suite_repeats"] = summarize_macros(report["jobs"])
        if not report["jobs"]:
            raise ValueError(f"No job receipts found under {args.root}")
        csv_path = args.output.with_suffix(".csv")
        if csv_path.exists():
            raise FileExistsError(csv_path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = [k for k in report["jobs"][0] if k != "result_files"]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["jobs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
