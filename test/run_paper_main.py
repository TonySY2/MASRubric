#!/usr/bin/env python3
"""Run the recovered Dynamic-MAS main method, isolated from the release runtime.

No model calls are made by --dry-run or --preflight. Each experiment records
fresh measurements; precomputed benchmark results are not bundled.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "paper" / "runtime"
CONFIG = ROOT / "configs" / "paper_main.json"
ROLES = ("selector", "reasoning", "supervisor", "embedding")
LOCKED_ENV = {
    "MASRUBRIC_BATCH_AUDIT_METRICS": "1",
    "MASRUBRIC_EXACT_SELECT_Q": "0",
    "MASRUBRIC_CHEAP_PRECHECK": "0",
    "MASRUBRIC_RANDOM_K_MIN": "0",
    "MASRUBRIC_RANDOM_K_MAX": "0",
    "MASRUBRIC_BATCH_AUDIT_MAX_TOKENS": "4000",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.split("\n") if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "test", "problems", "questions"):
            if isinstance(value.get(key), list):
                return value[key]
        return list(value.values())
    raise ValueError(f"Expected dataset records in {path}")


def iter_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from records(path)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("math_8b", "math_14b", "code_8b"))
    p.add_argument("--benchmark", default="all")
    p.add_argument("--method", choices=("masrubric", "baseline"), default="masrubric")
    p.add_argument("--assets-root", type=Path, default=ROOT / "paper" / "assets")
    p.add_argument("--in-file", type=Path)
    p.add_argument("--metric-pool-file", type=Path)
    p.add_argument("--embedding-cache-file", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "results")
    p.add_argument("--limit", type=positive)
    p.add_argument("--concurrency", type=positive, default=1)
    p.add_argument("--olympiad-python", type=Path, help="Interpreter with legacy latex2sympy2 for OlympiadBench.")
    p.add_argument("--allow-model-override", action="store_true", help="Explicitly allow model names unlike the historical role mapping (e.g. an offline smoke server).")
    p.add_argument("--timeout", type=positive, default=86400, help="Maximum seconds per benchmark subprocess.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--list", action="store_true")
    return p


def build_plan(args, config, benchmark, out_dir):
    spec = config["benchmarks"][benchmark]
    domain = "code" if args.suite == "code_8b" else "math"
    if spec["domain"] != domain:
        raise ValueError(f"{benchmark} does not belong to {args.suite}")
    assets = args.assets_root.resolve()
    data = (args.in_file or assets / spec["dataset"]).resolve()
    pool = (args.metric_pool_file or assets / config["pools"][domain]["metrics"]).resolve()
    cache = (args.embedding_cache_file or assets / config["pools"][domain]["embeddings"]).resolve()
    output = out_dir / benchmark
    script = RUNTIME / spec["script"]
    # The child receives one explicit runtime path. No release masrubric or
    # inherited experiment toggles may silently change the main-method preset.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MASRUBRIC_")}
    env["PYTHONPATH"] = str(RUNTIME)
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(LOCKED_ENV)
    env["MASRUBRIC_MATH_TEAM_VARIANT"] = "verifier" if domain == "math" else ""
    env["MASRUBRIC_PROFILE_AWARE_RETRIEVAL"] = "1" if domain == "code" else "0"
    env["MASRUBRIC_USAGE_LOG"] = str(output / "usage.jsonl")
    for role in ROLES:
        env[f"MASRUBRIC_{role.upper()}_API_KEY"] = os.environ.get(f"{role.upper()}_KEY", "EMPTY")
    result = output / ("result.jsonl" if domain == "code" else "result.json")
    # runpy plus an import assertion prevents an installed/release package from
    # taking precedence even when the runner lives several directories deep.
    bootstrap = ("import pathlib,runpy,sys; "
                 "r=pathlib.Path(sys.argv.pop(1)).resolve(); sys.path.insert(0,str(r)); "
                 "import masrubric; "
                 "assert pathlib.Path(masrubric.__file__).resolve().is_relative_to(r), 'Wrong runtime'; "
                 "s=sys.argv[1]; sys.path.insert(1,str(pathlib.Path(s).parent)); "
                 "sys.argv=sys.argv[1:]; runpy.run_path(s,run_name='__main__')")
    # Keep the venv's interpreter entrypoint: resolving its symlink would select
    # the base Python and silently lose the venv's grader dependencies.
    interpreter = str(args.olympiad_python.absolute()) if benchmark == "olympiad" and args.olympiad_python else sys.executable
    cmd = [interpreter, "-c", bootstrap, str(RUNTIME), str(script),
           "--in_file", str(data), "--out_file", str(result),
           "--log_file", str(output / "detailed.log"),
           "--metric_pool_file", str(pool), "--embedding_cache_file", str(cache),
           "--max_turns", "7", "--concurrency_limit", str(args.concurrency),
           "--pass_rate", "1.0" if domain == "code" else "0.6"]
    if domain == "code":
        cmd += ["--task", benchmark, "--force_direct_search", "--direct_k", "3"]
    else:
        cmd += ["--retrieve_p", "20", "--select_q", "5", "--retries_times", "3"]
    if args.method == "baseline":
        cmd.append("--baseline_only")
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    endpoints = {}
    missing_env = []
    for role in ROLES:
        endpoints[role] = {}
        for field in ("url", "model"):
            name = f"{role}_{field}".upper()
            value = os.environ.get(name)
            if not value:
                missing_env.append(name)
                value = f"<{name}>"
            cmd += [f"--{role}_{field}", value]
            endpoints[role][field] = value
    # Selector has a historical CLI key; other clients use role KEY variables.
    cmd += ["--selector_api_key", os.environ.get("SELECTOR_KEY", "EMPTY")]
    safe_cmd = cmd.copy()
    safe_cmd[safe_cmd.index("--selector_api_key") + 1] = "***"
    manifest = {
        "suite": args.suite, "benchmark": benchmark, "method": args.method,
        "runtime": str(RUNTIME), "reference_package": config["reference_package"],
        "source_boundary": config["source_boundary"], "python": interpreter,
        "scope": "recovered_dynamic_main" if args.suite != "math_14b" and args.method == "masrubric"
                 else "same_runtime_comparison_not_historical_row_certification",
        "dataset": str(data), "metric_pool": str(pool), "embedding_cache": str(cache),
        "benchmark_total": spec["total"], "limit": args.limit,
        "concurrency": args.concurrency, "models_and_endpoints": endpoints,
        "fixed_environment": {k: v for k, v in env.items() if k.startswith("MASRUBRIC_") and not k.endswith("_API_KEY")},
        "command": safe_cmd, "result_file": str(result),
        "sampling_note": "Preserved runtime sampling; no seed was recorded for the source runs.",
        "final_decision": spec["final_decision"],
        "historical_reference": spec["historical"],
        "model_override_allowed": args.allow_model_override,
    }
    return cmd, env, manifest, missing_env


def preflight(manifest, missing_env, args):
    errors = [f"Missing environment: {name}" for name in missing_env]
    if manifest["suite"] != "math_14b":
        mismatches = []
        for role, expected in manifest["historical_reference"]["model_roles"].items():
            actual = manifest["models_and_endpoints"][role]["model"].replace("\\", "/").rsplit("/", 1)[-1]
            if actual != expected:
                mismatches.append(f"{role}: expected {expected}, configured {actual}")
        manifest["model_name_mismatches"] = mismatches
        if mismatches and not args.allow_model_override:
            errors.append("Historical model mapping differs: " + "; ".join(mismatches) + ". Use --allow-model-override only for an intentional variant.")
    for key in ("dataset", "metric_pool", "embedding_cache"):
        if not Path(manifest[key]).is_file():
            errors.append(f"Missing {key}: {manifest[key]}")
    if Path(manifest["dataset"]).is_file():
        try:
            count, sample_ids = 0, []
            for i, row in enumerate(iter_records(manifest["dataset"])):
                count += 1
                if args.limit is None or i < args.limit:
                    sample_ids.append(str(next((row[k] for k in ("id", "task_id", "problem_id", "question_id")
                                                if isinstance(row, dict) and k in row), i)))
            manifest["input_records"] = count
            if count != manifest["benchmark_total"]:
                errors.append(f"Dataset has {count} records; expected full benchmark {manifest['benchmark_total']}")
            manifest["selected_samples"] = len(sample_ids)
            manifest["sample_ids"] = sample_ids
        except (ValueError, TypeError) as exc:
            errors.append(f"Invalid dataset: {exc}")
    if Path(manifest["metric_pool"]).is_file() and Path(manifest["embedding_cache"]).is_file():
        try:
            pool = read_json(manifest["metric_pool"])
            vectors = {r["name"]: len(r["vector"]) for r in iter_records(manifest["embedding_cache"])}
            missing = [r["name"] for r in pool if r["name"] not in vectors]
            if missing:
                errors.append(f"Embedding cache misses {len(missing)} pool indicators")
            dims = {vectors[r["name"]] for r in pool if r["name"] in vectors}
            if len(dims) != 1 or 0 in dims:
                errors.append("Embedding vectors have inconsistent or zero dimensions")
            manifest["indicator_count"] = len(pool)
            if manifest["suite"] == "code_8b":
                profiles = sum(isinstance(r.get("match_profile"), dict) for r in pool)
                usable = sum(isinstance(r.get("match_profile"), dict) and
                             all(r["match_profile"].get(k) for k in ("interface_shape", "io_contract")) for r in pool)
                manifest["pool_profile_coverage"] = {"total": len(pool), "with_match_profile": profiles,
                                                     "missing_match_profile": len(pool) - profiles,
                                                     "with_interface_and_io": usable}
                if usable < len(pool):
                    manifest.setdefault("warnings", []).append(
                        f"{len(pool) - usable} indicators lack complete match_profile fields; the preserved runtime defaults missing fields to any. "
                        "Enabling profile-aware retrieval does not prove those indicators can be filtered by interface.")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"Invalid pool/cache: {exc}")
    return errors


def grader_preflight(manifest, env):
    code = None
    if manifest["benchmark"] == "olympiad":
        code = "from grader import math_equal; assert math_equal('0.5', r'\\frac{1}{2}'), 'Legacy math grader probe failed'"
    elif manifest["benchmark"] in ("olymmath_easy", "olymmath_hard"):
        code = "from math_verify import parse,verify; assert verify(parse('$0.5$'), parse(r'$\\frac{1}{2}$')), 'Math verifier probe failed'"
    if code is None:
        return []
    try:
        result = subprocess.run([manifest["python"], "-c", code], cwd=RUNTIME, env=env,
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"Grader preflight failed: {exc}"]
    if result.returncode:
        return ["Required historical grader unavailable; refusing silent fallback: " + result.stderr[-1200:]]
    manifest["grader_probe"] = "passed"
    return []


def result_summary(manifest, returncode):
    result = Path(manifest["result_file"])
    if result.is_file() and result.suffix != ".jsonl":
        value = read_json(result)
        rows = [dict(row, id=row.get("id", key)) for key, row in value.items()] if isinstance(value, dict) else value
    else:
        rows = records(result) if result.is_file() else []
    ids = [str(row.get("id", i)) for i, row in enumerate(rows)]
    duplicates = len(ids) != len(set(ids))
    def correct(row):
        if manifest["benchmark"] == "aqua":
            # The historical AQuA runner writes the extracted option directly,
            # without the boolean grading field used by other runners.
            gold, pred = row.get("answer"), row.get("hypothesis")
            if isinstance(gold, list):
                gold = gold[0] if gold else ""
            return bool(str(gold or "").strip()) and str(gold).strip().upper() == str(pred or "").strip().upper()
        return row.get("is_correct", row.get("is_solved", False)) is True
    solved = sum(correct(row) for row in rows)
    expected = manifest["selected_samples"]
    complete = returncode == 0 and len(rows) == expected and not duplicates
    return {
        "benchmark": manifest["benchmark"], "returncode": returncode,
        "completed_samples": len(rows), "expected_samples": expected,
        "correct": solved, "missing_samples": max(0, expected - len(rows)),
        "duplicate_sample_ids": duplicates, "complete": complete,
        "accuracy_percent": 100 * solved / expected if expected else None,
        "benchmark_total": manifest["benchmark_total"],
        "scope": "smoke_subset" if manifest["limit"] else "full_benchmark",
        "denominator_policy": "Selected/full benchmark; missing outputs count as incorrect.",
    }


def regrade_code(manifest, summary, env, timeout):
    """Use the source experiment's separate paper grader, never stored flags."""
    task_dir = Path(manifest["result_file"]).parent
    report = task_dir / "paper_grader.json"
    cmd = [sys.executable, str(RUNTIME / "paper_code_eval.py"), "--task", manifest["benchmark"],
           "--result-file", manifest["result_file"], "--dataset-file", manifest["dataset"], "--json"]
    with report.open("w", encoding="utf-8") as output, (task_dir / "grader.log").open("w", encoding="utf-8") as errors:
        try:
            process = subprocess.run(cmd, cwd=RUNTIME, env=env, stdout=output, stderr=errors, timeout=timeout)
        except subprocess.TimeoutExpired:
            summary.update(complete=False, grader_error="timeout")
            return
    if process.returncode:
        summary.update(complete=False, grader_error=f"exit {process.returncode}")
        return
    try:
        graded = read_json(report)
        summary["correct"] = graded["correct"]
        summary["accuracy_percent"] = 100 * graded["correct"] / summary["expected_samples"]
        summary["grader"] = "paper_code_eval.py (re-executed tests; stored flags not trusted)"
        summary["grader_matched_records"] = graded["matched_records"]
        summary["complete"] = summary["complete"] and graded["matched_records"] == summary["completed_samples"]
    except (ValueError, KeyError, TypeError) as exc:
        summary.update(complete=False, grader_error=str(exc))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    config = read_json(CONFIG)
    if args.list:
        for domain in ("math", "code"):
            print(domain + ": " + ", ".join(k for k, v in config["benchmarks"].items() if v["domain"] == domain))
        print("Methods: masrubric; baseline (same-runtime control). Fixed/other paper baselines are outside this entrypoint.")
        return 0
    if not args.suite:
        p.error("--suite is required")
    domain = "code" if args.suite == "code_8b" else "math"
    chosen = [k for k, v in config["benchmarks"].items() if v["domain"] == domain] if args.benchmark == "all" else [args.benchmark]
    if args.in_file and len(chosen) != 1:
        p.error("--in-file requires one benchmark")
    if any(k not in config["benchmarks"] for k in chosen):
        p.error("Unknown benchmark; use --list")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output_dir.resolve() / args.suite / args.method / stamp
    plans, all_errors = [], []
    for benchmark in chosen:
        try:
            plan = build_plan(args, config, benchmark, out)
        except ValueError as exc:
            p.error(str(exc))
        cmd, env, manifest, missing = plan
        if args.dry_run:
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        else:
            errors = preflight(manifest, missing, args)
            if not errors:
                errors += grader_preflight(manifest, env)
            all_errors += [f"{benchmark}: {e}" for e in errors]
            if args.preflight:
                print(json.dumps({"benchmark": benchmark, "errors": errors, "manifest": manifest}, ensure_ascii=False, indent=2))
        plans.append(plan)
    if all_errors:
        print("Preflight failed:\n" + "\n".join(all_errors), file=sys.stderr)
        return 2
    if args.dry_run or args.preflight:
        return 0
    summaries = []
    for cmd, env, manifest, _ in plans:
        task_dir = Path(manifest["result_file"]).parent
        task_dir.mkdir(parents=True, exist_ok=False)
        (task_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Running {manifest['benchmark']}: {task_dir}", flush=True)
        with (task_dir / "process.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(cmd, cwd=RUNTIME, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                rc = completed.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        try:
            summary = result_summary(manifest, rc)
            if args.suite == "code_8b" and Path(manifest["result_file"]).is_file():
                regrade_code(manifest, summary, env, args.timeout)
        except (ValueError, TypeError) as exc:
            summary = {"benchmark": manifest["benchmark"], "complete": False, "error": str(exc)}
        (task_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    complete = all(s["complete"] for s in summaries)
    suite_summary = {"complete": complete, "tasks": summaries,
                     "macro_accuracy_percent": sum(s["accuracy_percent"] for s in summaries) / len(summaries)
                     if complete else None,
                     "scope": "smoke_subset" if args.limit else "full_benchmark"}
    (out / "summary.json").write_text(json.dumps(suite_summary, indent=2), encoding="utf-8")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
