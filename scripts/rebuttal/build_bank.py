#!/usr/bin/env python3
"""Plan or run failure collection, criterion deduplication, and embeddings."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare_assets import ROOT, iter_records, validate_bank, validate_cache

TRAIN = ROOT / "train"
EMBED_SCRIPT = ROOT / "test/metrics_pool/two_pool/embed_metrics-trigger.py"
BOOTSTRAP = """import json,os,runpy,sys
script=sys.argv[1]
sys.path.insert(0,os.environ['MASRUBRIC_BANK_IMPORT_ROOT'])
args=json.loads(os.environ.pop('MASRUBRIC_BANK_ARGS'))
for flag,var in json.loads(os.environ.pop('MASRUBRIC_BANK_SECRET_ARGS')):
    args.extend([flag,os.environ.get(var,'EMPTY')])
sys.argv=[script]+args
runpy.run_path(script,run_name='__main__')
"""


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def raw_trajectory_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Expected a nonempty task-to-trajectory JSON object")
    mined = failures = 0
    for row in data.values():
        if not isinstance(row, dict):
            raise ValueError("Each task record must be an object")
        failures += row.get("is_correct") is False
        trajectories = row.get("trajectories", {})
        if isinstance(trajectories, dict):
            for steps in trajectories.values():
                if isinstance(steps, list):
                    for step in steps:
                        if isinstance(step, dict):
                            mined += len(step.get("generated_adversarial_metrics", []))
    return {"tasks": len(data), "failed_tasks": failures, "raw_criterion_occurrences": mined}


def make_plan(args: argparse.Namespace) -> dict:
    out = args.output_dir.resolve()
    inputs = [p.resolve() for p in args.input_file]
    if args.stage.startswith("collect-"):
        domain = args.stage.split("-", 1)[1]
        if not inputs:
            inputs = [TRAIN / ("project_datasets/MATH/processed_math/train.jsonl" if domain == "math"
                               else "project_datasets/aqua/train.jsonl")]
        if len(inputs) != 1:
            raise ValueError("Collection accepts one training file")
        script = TRAIN / f"experiments/{domain}/run_{domain}.py"
        params = ["--in_file", str(inputs[0]), "--out_file", str(out / "trajectories.json"),
                  "--log_file", str(out / "trajectory.log"), "--max_turns", str(args.max_turns)]
        if args.limit is not None:
            params += ["--limit", str(args.limit)]
        required, secrets = [], []
        for role in ("SELECTOR", "REASONING", "SUPERVISOR", "EMBEDDING"):
            for kind in ("URL", "MODEL"):
                name = f"{role}_{kind}"
                required.append(name)
                params += [f"--{role.lower()}_{kind.lower()}", os.environ.get(name, f"<{name}>")]
            secrets.append((f"--{role.lower()}_key", f"{role}_KEY"))
        outputs = ["trajectories.json", "trajectory.log"]
    elif args.stage == "deduplicate":
        if len(inputs) != 1:
            raise ValueError("deduplicate needs one merged trajectory file")
        script = TRAIN / "Extraction-deduplication-embedding.py"
        params = ["--input_data_file", str(inputs[0]), "--output_metrics_file", str(out / "bank.json"),
                  "--output_embedding_file", str(out / "embeddings.jsonl")]
        required = ["SUPERVISOR_URL", "SUPERVISOR_MODEL", "EMBEDDING_URL", "EMBEDDING_MODEL"]
        secrets, outputs = [], ["bank.json", "embeddings.jsonl"]
    elif args.stage == "embed":
        if len(inputs) != 1:
            raise ValueError("embed needs one criterion bank JSON file")
        script = EMBED_SCRIPT
        params = ["--input_file", str(inputs[0]), "--output_cache_file", str(out / "embeddings.jsonl")]
        required = ["EMBEDDING_URL", "EMBEDDING_MODEL"]
        secrets, outputs = [], ["embeddings.jsonl"]
    else:
        if not inputs:
            raise ValueError("merge-trajectories needs one or more --input-file arguments")
        script, params, required, secrets = None, [], [], []
        outputs = ["trajectories.json"]
    return {"stage": args.stage, "inputs": [str(p) for p in inputs], "output_dir": str(out),
            "script": str(script) if script else None, "arguments": params,
            "required_environment": required, "secret_environment": [var for _, var in secrets],
            "output_files": outputs, "_secret_args": secrets}


def run_plan(plan: dict, args: argparse.Namespace) -> dict:
    out = Path(plan["output_dir"])
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    for name in plan["inputs"]:
        if not Path(name).is_file():
            raise FileNotFoundError(f"Missing input: {name}")
    missing = [name for name in plan["required_environment"] if not os.environ.get(name)]
    if missing:
        raise ValueError("Set environment variables: " + ", ".join(missing))
    inputs = [Path(name) for name in plan["inputs"]]
    before = {}
    if args.stage == "deduplicate":
        before = raw_trajectory_summary(inputs[0])
        if not before["raw_criterion_occurrences"]:
            raise ValueError("Input has no generated_adversarial_metrics; run failure collection/mining first")
    elif args.stage == "embed":
        before, _ = validate_bank(inputs[0])
        if before["errors"]:
            raise ValueError("; ".join(before["errors"]))
    elif args.stage.startswith("collect-"):
        records = sum(1 for _ in iter_records(inputs[0]))
        if records == 0:
            raise ValueError("Training dataset is empty")
        before = {"input_records": records, "selected_records": min(records, args.limit) if args.limit else records}
    else:
        before = {"sources": [raw_trajectory_summary(path) for path in inputs]}
    out.mkdir(parents=True, exist_ok=False)
    public_plan = {key: value for key, value in plan.items() if not key.startswith("_")}
    # URLs can contain credentials. The on-disk plan records option names and
    # environment-variable names, never endpoint values or key values.
    public_plan["arguments"] = [item if not (i and plan["arguments"][i-1].endswith("_url"))
                                else "<environment>" for i, item in enumerate(plan["arguments"])]
    (out / "plan.json").write_text(json.dumps(public_plan, indent=2) + "\n", encoding="utf-8")
    if args.stage == "merge-trajectories":
        merged = {}
        for index, path in enumerate(inputs):
            for task_id, row in json.loads(path.read_text(encoding="utf-8-sig")).items():
                merged[f"source{index}:{task_id}"] = row
        (out / "trajectories.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        returncode = 0
    else:
        env = os.environ.copy()
        env.update(PYTHONPATH=str(TRAIN), PYTHONIOENCODING="utf-8",
                   MASRUBRIC_BANK_IMPORT_ROOT=str(TRAIN),
                   MASRUBRIC_BANK_ARGS=json.dumps(plan["arguments"]),
                   MASRUBRIC_BANK_SECRET_ARGS=json.dumps(plan["_secret_args"]))
        with (out / "process.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-u", "-c", BOOTSTRAP, plan["script"]],
                                  cwd=TRAIN, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        returncode = proc.returncode
    errors, after = [], {}
    if returncode:
        errors.append(f"Child process exited {returncode}; inspect process.log")
    try:
        if args.stage in ("embed", "deduplicate"):
            bank = out / "bank.json" if args.stage == "deduplicate" else inputs[0]
            bank_report, names = validate_bank(bank)
            cache_report = validate_cache(out / "embeddings.jsonl", names)
            after = {"bank": bank_report, "cache": cache_report}
            errors.extend(bank_report["errors"] + cache_report["errors"])
        else:
            after = raw_trajectory_summary(out / "trajectories.json")
            expected = before.get("selected_records")
            if expected is not None and after["tasks"] != expected:
                errors.append(f"Collection produced {after['tasks']} of {expected} selected tasks")
    except (OSError, ValueError, TypeError) as exc:
        errors.append(str(exc))
    report = {"stage": args.stage, "status": "complete" if not errors else "failed",
              "before": before, "after": after, "errors": errors, "returncode": returncode}
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect-math", "collect-aqua", "merge-trajectories", "deduplicate", "embed"))
    parser.add_argument("--input-file", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=positive, help="Collection prefix size; omit to use the complete training input.")
    parser.add_argument("--max-turns", type=positive, default=7)
    parser.add_argument("--execute", action="store_true", help="Run the plan. Collection/mining/embedding makes model calls.")
    args = parser.parse_args(argv)
    plan = make_plan(args)
    if not args.execute:
        view = {key: value for key, value in plan.items() if not key.startswith("_")}
        # A dry run is shareable even if a private endpoint embeds authentication.
        view["arguments"] = [item if not (i and plan["arguments"][i-1].endswith("_url"))
                             else "<environment>" for i, item in enumerate(plan["arguments"])]
        print(json.dumps(view, ensure_ascii=False, indent=2))
        return 0
    report = run_plan(plan, args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Bank preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
