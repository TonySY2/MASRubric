#!/usr/bin/env python3
"""Plan and run isolated rebuttal experiments. Planning makes no model calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rebuttal_matrix.json"
PAPER = ROOT / "configs/paper_main.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def make_plan(args):
    matrix = json.loads(CONFIG.read_text(encoding="utf-8"))
    paper = json.loads(PAPER.read_text(encoding="utf-8"))
    family = matrix["families"][args.family]
    domain = "code" if args.suite == "code_8b" else "math"
    if domain not in family["domains"]:
        raise ValueError(f"{args.family} supports {family['domains']}, not {domain}")
    benchmarks = [k for k, v in paper["benchmarks"].items() if v["domain"] == domain]
    if args.benchmarks != "all":
        requested = args.benchmarks.split(",")
        if len(requested) != len(set(requested)) or any(k not in benchmarks for k in requested):
            raise ValueError("Use unique benchmark ids belonging to the selected suite")
        benchmarks = requested
    output = args.output_dir.absolute()
    jobs = []
    release_config = None
    if family["runtime"] != "paper":
        release_config = json.loads((ROOT / "configs/rebuttal_experiments.json").read_text(encoding="utf-8"))
    for benchmark in benchmarks:
        spec = paper["benchmarks"][benchmark]
        for repeat in range(1, args.repeats + 1):
            # Adjacent control and treatment jobs keep their scheduling comparable.
            for method in family["methods"]:
                name = method.format(domain=domain)
                job_id = f"{args.suite}__{benchmark}__{name}__r{repeat:02d}"
                destination = output / job_id
                if family["runtime"] == "paper":
                    reference_method = name if name in ("baseline", "masrubric") else "masrubric"
                    command = [sys.executable, str(ROOT / "test/run_paper_main.py"),
                               "--suite", args.suite, "--benchmark", benchmark,
                               "--method", reference_method, "--output-dir", str(destination),
                               "--concurrency", str(args.concurrency)]
                    if reference_method != name:
                        command += ["--variant-name", name] + matrix["reference_variants"][name]
                    if args.assets_root:
                        command += ["--assets-root", str(args.assets_root.absolute())]
                    if args.olympiad_python:
                        command += ["--olympiad-python", str(args.olympiad_python.absolute())]
                    if args.allow_model_override:
                        command.append("--allow-model-override")
                else:
                    command = [sys.executable, str(ROOT / "test/run_release_experiment.py"),
                               "--config", str(ROOT / "configs/rebuttal_experiments.json"),
                               "--benchmark", benchmark, "--method", name,
                               "--model-profile", args.suite,
                               "--framework", family.get("framework", "dynamic"),
                               "--output-dir", str(destination)]
                    if args.assets_root:
                        command += ["--in-file", str(args.assets_root.absolute() / spec["dataset"])]
                        # Pool-specific cache overrides are taken from environment for
                        # custom / non-deduplicated banks, as in the base launcher.
                    if benchmark == "olympiad" and args.olympiad_python:
                        command[0] = str(args.olympiad_python.absolute())
                if args.limit:
                    command += ["--limit", str(args.limit)]
                if args.assets_root:
                    dataset_file = args.assets_root.absolute() / spec["dataset"]
                elif release_config is not None:
                    item = release_config["benchmarks"][benchmark]
                    value = os.environ.get(item.get("input_file_env", "")) or item.get("input_file")
                    dataset_file = Path(value) if value else None
                    if dataset_file is not None and not dataset_file.is_absolute():
                        dataset_file = ROOT / dataset_file
                else:
                    dataset_file = ROOT / "paper/assets" / spec["dataset"]
                jobs.append({"id": job_id, "benchmark": benchmark, "method": name, "runtime": family["runtime"],
                             "suite": args.suite, "family": args.family,
                             "repeat": repeat, "repeat_kind": "fresh_stochastic_run_no_seed_contract",
                             "expected": min(spec["total"], args.limit or spec["total"]),
                             "full_benchmark_total": spec["total"],
                             "dataset_file": str(dataset_file) if dataset_file is not None else None,
                             "output_dir": str(destination), "command": command})
    return {"schema_version": 1, "family": args.family, "suite": args.suite,
            "runtime": family["runtime"], "interpretation": family["interpretation"],
            "created_at": datetime.now(timezone.utc).isoformat(), "jobs": jobs,
            "endpoint_variables": [f"{role}_{field}" for role in
                                   ("REASONING", "SUPERVISOR", "SELECTOR", "EMBEDDING")
                                   for field in ("URL", "MODEL", "KEY")],
            "notes": ["Commands inherit secrets only at execution; plans contain none.",
                      "Run separate suite plans after configuring the required model roles.",
                      "Repeated runs are never renamed independent random seeds."]}


def stop_child(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def validate_resume(job, previous, summarize_job):
    keys = ("id", "command", "benchmark", "method", "expected", "full_benchmark_total",
            "suite", "runtime", "family", "repeat", "dataset_file", "output_dir")
    changed = [key for key in keys if job.get(key) != previous.get(key)]
    if changed:
        raise ValueError("Existing receipt belongs to a different job configuration: " + ", ".join(changed))
    if previous.get("status") != "completed":
        return False
    if not summarize_job(previous)["complete"]:
        raise ValueError("Completed receipt no longer has complete verified results; preserve it and create a new run")
    return True


def execute(plan, args):
    # Plans are local trusted command artifacts. Restrict this entry point to
    # generated project launchers, even if the JSON was edited later.
    allowed = {(ROOT / "test/run_paper_main.py").resolve(),
               (ROOT / "test/run_release_experiment.py").resolve()}
    from summarize import summarize_job
    failed = 0
    for job in plan["jobs"]:
        cmd = job["command"]
        if len(cmd) < 2 or Path(cmd[1]).resolve() not in allowed:
            raise ValueError("Plan contains a launcher outside this repository; regenerate it here")
        if args.action == "check":
            result = subprocess.run(cmd + ["--dry-run"], cwd=ROOT, capture_output=True, text=True, timeout=args.timeout)
            print(f"{job['id']}: {'command_ok' if result.returncode == 0 else 'invalid'}")
            failed += result.returncode != 0
            if result.returncode:
                print(result.stderr[-2000:])
            continue
        destination = Path(job["output_dir"])
        receipt = destination / "job.json"
        if destination.exists():
            if args.resume and receipt.exists():
                previous = json.loads(receipt.read_text(encoding="utf-8"))
                if validate_resume(job, previous, summarize_job):
                    print(f"{job['id']}: already completed")
                    continue
            raise FileExistsError(f"Existing job directory: {destination}; preserve it and use a new plan/output directory")
        destination.mkdir(parents=True)
        safe_roles = {name: os.environ.get(name) for name in plan["endpoint_variables"] if not name.endswith("_KEY")}
        state = dict(job, status="running", started_at=datetime.now(timezone.utc).isoformat(),
                     models_and_endpoints=safe_roles, runtime=plan["runtime"], family=plan["family"])
        write_json(receipt, state)
        started = time.monotonic()
        process = None
        print(f"{job['id']}: running", flush=True)
        try:
            with (destination / "launcher.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=os.name == "posix")
                code = process.wait(timeout=args.timeout)
            state.update(returncode=code, status="completed" if code == 0 else "failed")
            if code == 0:
                summary = summarize_job(state)
                state["result_summary"] = summary
                if not summary["complete"]:
                    state["status"] = "incomplete"
        except subprocess.TimeoutExpired:
            if process is not None:
                stop_child(process)
            state.update(status="timeout", failure_kind="job_timeout")
        except KeyboardInterrupt:
            if process is not None:
                stop_child(process)
            state["status"] = "interrupted"
            raise
        except Exception as exc:
            if process is not None and process.poll() is None:
                stop_child(process)
            state.update(status="failed", failure_kind=type(exc).__name__, failure_message=str(exc)[:1000])
        finally:
            state["elapsed_seconds"] = time.monotonic() - started
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(receipt, state)
        print(f"{job['id']}: {state['status']}", flush=True)
        if state["status"] != "completed":
            failed += 1
            if not args.keep_going:
                break
    return int(failed > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["list", "plan", "check", "run"])
    p.add_argument("--family", default="main")
    p.add_argument("--suite", choices=["math_8b", "math_14b", "code_8b"], default="math_8b")
    p.add_argument("--benchmarks", default="all")
    p.add_argument("--repeats", type=positive, default=1)
    p.add_argument("--limit", type=positive)
    p.add_argument("--concurrency", type=positive, default=1)
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/rebuttal")
    p.add_argument("--plan", type=Path, default=ROOT / "runs/rebuttal-plan.json")
    p.add_argument("--assets-root", type=Path)
    p.add_argument("--olympiad-python", type=Path)
    p.add_argument("--allow-model-override", action="store_true")
    p.add_argument("--timeout", type=positive, default=86400)
    p.add_argument("--resume", action="store_true", help="Skip completed jobs; never overwrite incomplete runs")
    p.add_argument("--keep-going", action="store_true")
    args = p.parse_args()
    if args.action == "list":
        print(CONFIG.read_text(encoding="utf-8"))
        return 0
    if args.action == "plan":
        if args.plan.exists():
            raise FileExistsError(f"Preserve the existing plan: {args.plan}")
        plan = make_plan(args)
        write_json(args.plan, plan)
        print(f"Prepared {len(plan['jobs'])} jobs: {args.plan}")
        return 0
    return execute(json.loads(args.plan.read_text(encoding="utf-8")), args)


if __name__ == "__main__":
    raise SystemExit(main())
