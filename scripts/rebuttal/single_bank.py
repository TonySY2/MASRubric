#!/usr/bin/env python3
"""One recovered solver with the original bank/retriever/auditor; measured cost."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "paper/runtime"
MODELS = {"reasoning": "Qwen3-8B", "supervisor": "Qwen3-8B",
          "embedding": "Qwen3-Embedding-8B", "selector": "Qwen3.5-9B"}
DATASETS = {"math500": "datasets/math500/test_ids.jsonl", "aqua": "datasets/aqua/aqua_id.jsonl"}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def task_text(row, benchmark):
    if benchmark == "math500":
        return row.get("problem", row.get("question", ""))
    # Exact task formatting from reference project_datasets.aqua_dataset.
    return row["question"] + " Choices:" + "".join(" " + option for option in row["options"])


def account_usage(events):
    totals = {"llm_total_tokens": 0, "embedding_total_tokens": 0,
              "llm_prompt_tokens": 0, "llm_completion_tokens": 0, "unknown_calls": 0}
    for event in events:
        usage = event.get("usage")
        embedding = event.get("role") == "embedding"
        fields = ("prompt_tokens", "total_tokens") if embedding else ("prompt_tokens", "completion_tokens", "total_tokens")
        if event.get("status") != "ok" or not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in fields):
            totals["unknown_calls"] += 1
            continue
        if not embedding and usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
            totals["unknown_calls"] += 1
            continue
        if embedding:
            totals["embedding_total_tokens"] += usage["total_tokens"]
        else:
            totals["llm_total_tokens"] += usage["total_tokens"]
            totals["llm_prompt_tokens"] += usage["prompt_tokens"]
            totals["llm_completion_tokens"] += usage["completion_tokens"]
    return {**totals, "complete": totals["unknown_calls"] == 0, "logical_calls": len(events)}


def instrument(client, role, path, events, asynchronous=True):
    resource = client.embeddings if role == "embedding" else client.chat.completions
    original = resource.create

    def record(kwargs, result=None, error=None, started=0):
        event = {"role": role, "status": "error" if error else "ok",
                 "elapsed_seconds": time.monotonic() - started,
                 "request": {k: v for k, v in kwargs.items() if k in
                             ("model", "messages", "input", "temperature", "max_tokens", "extra_body")}}
        if error:
            event["error_type"] = type(error).__name__
        else:
            payload = result.model_dump()
            event.update(response=payload, usage=payload.get("usage"))
        events.append(event)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    async def async_create(**kwargs):
        started = time.monotonic()
        try:
            response = await original(**kwargs)
        except BaseException as exc:
            record(kwargs, error=exc, started=started)
            raise
        record(kwargs, result=response, started=started)
        return response

    def sync_create(**kwargs):
        started = time.monotonic()
        try:
            response = original(**kwargs)
        except BaseException as exc:
            record(kwargs, error=exc, started=started)
            raise
        record(kwargs, result=response, started=started)
        return response

    resource.create = async_create if asynchronous else sync_create


def load_bank(pool, cache):
    from prepare_assets import validate_bank, validate_cache
    import numpy as np
    report, names = validate_bank(pool)
    cache_report = validate_cache(cache, names)
    if report["errors"] or cache_report["errors"]:
        raise ValueError("Invalid bank/cache: " + "; ".join(report["errors"] + cache_report["errors"]))
    metrics = json.loads(pool.read_text(encoding="utf-8"))
    vectors = {}
    with cache.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                vectors[item["name"]] = item["vector"]
    return metrics, np.array([vectors[m["name"]] for m in metrics], dtype=np.float32)


async def execute_sample(row, args, config, directory, metrics, vectors):
    from autogen_core import CancellationToken
    from autogen_agentchat.messages import TextMessage
    from masrubric.agents.supervisor_reasoning_pick_metric import Supervisor
    from masrubric.agents.math_solver_math500 import MathSolverMath500
    from masrubric.agents.math_solver_aqua import MathSolverAqua
    from masrubric.agents.math_grader import MathGrader
    from project_datasets.aqua_dataset import aqua_get_predict

    directory.mkdir(parents=True, exist_ok=False)
    os.environ["MASRUBRIC_USAGE_LOG"] = str(directory / "usage.jsonl")
    os.environ["MASRUBRIC_MECHANISM_TRACE"] = str(directory / "mechanism.jsonl")
    endpoints = config["endpoints"]
    key = lambda role: os.environ.get(role.upper() + "_KEY", "EMPTY")
    supervisor = Supervisor(model=endpoints["supervisor"]["model"], api_key=key("supervisor"),
        base_url=endpoints["supervisor"]["url"], domain=args.benchmark,
        embedding_api_key=key("embedding"), embedding_api_base=endpoints["embedding"]["url"],
        embedding_model=endpoints["embedding"]["model"], preloaded_metrics=metrics,
        preloaded_embeddings=vectors, retrieve_p=20, select_q=5, pass_rate=.6,
        prune_flag=args.method != "vanilla", lock_metrics_after_first_round=False)
    cls = MathSolverMath500 if args.benchmark == "math500" else MathSolverAqua
    agent = cls(domain=args.benchmark, name="SingleSolver", role="Math Solver", message_history=[],
                model=endpoints["reasoning"]["model"], api_key=key("reasoning"),
                base_url=endpoints["reasoning"]["url"], supervisor=supervisor,
                reflection_time=3)
    events = []
    for client, role, asynchronous in ((agent._model_client, "reasoning", True),
            (supervisor._model_client, "supervisor", True), (supervisor.embedding_client, "embedding", False)):
        instrument(client, role, directory / "api_calls.jsonl", events, asynchronous)
    result = {"id": str(row["id"]), "question": task_text(row, args.benchmark),
              "status": "unknown", "is_correct": None, "method": args.method}
    start = time.monotonic()
    try:
        response = await asyncio.wait_for(agent.on_messages(
            [TextMessage(source="user", content=result["question"])], CancellationToken()), timeout=args.timeout)
        scores = supervisor.get_scores({"SingleSolver": "Math Solver"})
        score = scores[response.chat_message.id]
        candidate = response.chat_message.content
        withheld = score["is_pruned"]
        if args.benchmark == "math500":
            candidate_correct = MathGrader.check_correctness(candidate, row.get("solution", row.get("answer", "")))
        else:
            candidate_correct = aqua_get_predict(candidate) == str(row["correct"]).strip()
        result.update(status="abstained" if withheld else "completed", candidate=candidate,
                      answer=None if withheld else candidate, candidate_correct=bool(candidate_correct),
                      is_correct=False if withheld else bool(candidate_correct), scores=scores,
                      action=score["action"])
    except (Exception, asyncio.CancelledError) as exc:
        result.update(error_type=type(exc).__name__, scores=supervisor.get_scores({"SingleSolver": "Math Solver"}))
    finally:
        result.update(usage=account_usage(events), elapsed_seconds=time.monotonic() - start,
                      sample_directory=str(directory))
        write_json(directory / "result.json", result)
        await agent._model_client.close()
        await supervisor._model_client.close()
        supervisor.embedding_client.close()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=DATASETS, required=True)
    p.add_argument("--method", choices=("vanilla", "full", "revise-only"), required=True)
    p.add_argument("--assets-root", type=Path, default=ROOT / "paper/assets")
    p.add_argument("--in-file", type=Path)
    p.add_argument("--metric-pool-file", type=Path)
    p.add_argument("--embedding-cache-file", type=Path)
    p.add_argument("--ids", help="Exact comma-separated problem IDs; omit for all")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--allow-model-override", action="store_true")
    p.add_argument("--execute", action="store_true", help="Default only prepares/displays the plan")
    args = p.parse_args(argv)
    if args.timeout <= 0:
        p.error("--timeout must be positive")
    dataset = args.in_file or args.assets_root / DATASETS[args.benchmark]
    pool = args.metric_pool_file or args.assets_root / "pools/math/deduped-mixed_metrics_two_pool.json"
    cache = args.embedding_cache_file or args.assets_root / "pools/math/deduped-mixed_two_pool-trigger.jsonl"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").split("\n") if line.strip()]
    indexed = {str(row["id"]): row for row in rows}
    ids = args.ids.split(",") if args.ids else list(indexed)
    if len(indexed) != len(rows) or len(set(ids)) != len(ids) or any(i not in indexed for i in ids) or not ids:
        raise ValueError("Use unique existing problem IDs from a nonempty dataset")
    endpoints = {role: {"model": os.environ.get(role.upper()+"_MODEL", MODELS[role]),
                        "url": os.environ.get(role.upper()+"_URL", "")}
                 for role in ("reasoning", "supervisor", "embedding")}
    mismatch = [role for role in endpoints if endpoints[role]["model"] != MODELS[role]]
    config = {"protocol": "single_reference_solver_one_answer_episode", "benchmark": args.benchmark,
              "method": args.method, "dataset": str(dataset.resolve()), "sample_ids": ids,
              "full_benchmark_total": len(rows), "metric_pool": str(pool.resolve()),
              "embedding_cache": str(cache.resolve()), "endpoints": endpoints,
              "group_chat_selector": {"reference_model": MODELS["selector"], "used": False},
              "criterion_reranker": "original Supervisor using supervisor model",
              "revision_budget": 3, "retrieve_p": 20, "select_q": 5, "pass_rate": .6,
              "runtime": str(RUNTIME.resolve()), "solver_role": "Math Solver",
              "sample_timeout_seconds": args.timeout, "model_override_allowed": args.allow_model_override,
              "no_final_decision_model": True, "fallback": "none; terminal withholding is abstention",
              "budget_policy": "measured usage, no hard total-token cap", "model_mismatches": mismatch}
    if not args.execute:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output_dir.mkdir(parents=True)
    config["started_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / "manifest.json", config)
    results = {}
    try:
        if mismatch and not args.allow_model_override:
            raise ValueError("Unexpected model names for roles: " + ", ".join(mismatch))
        if any(not value["url"] for value in endpoints.values()):
            raise ValueError("REASONING_URL, SUPERVISOR_URL and EMBEDDING_URL are required")
        # Eliminate inherited experimental toggles before loading the reference.
        for name in list(os.environ):
            if name.startswith("MASRUBRIC_"):
                del os.environ[name]
        os.environ.update(MASRUBRIC_BATCH_AUDIT_METRICS="1", MASRUBRIC_EXACT_SELECT_Q="0",
            MASRUBRIC_CHEAP_PRECHECK="0", MASRUBRIC_PROFILE_AWARE_RETRIEVAL="0",
            MASRUBRIC_BATCH_AUDIT_MAX_TOKENS="4000", MASRUBRIC_MATH_TEAM_VARIANT="verifier",
            MASRUBRIC_WITHHOLDING_POLICY="broadcast" if args.method == "revise-only" else "withhold")
        sys.path.insert(0, str(RUNTIME))
        import masrubric
        if not Path(masrubric.__file__).resolve().is_relative_to(RUNTIME):
            raise RuntimeError("Wrong masrubric runtime already imported; run this CLI in a fresh process")
        metrics, vectors = load_bank(pool, cache)
        config["bank_count"] = len(metrics)
        write_json(args.output_dir / "manifest.json", config)
        async def run_all():
            for index, identity in enumerate(ids):
                result = await execute_sample(indexed[identity], args, config,
                                              args.output_dir / "samples" / f"{index:05d}", metrics, vectors)
                results[identity] = result
                write_json(args.output_dir / "result.json", results)
                if result["status"] == "unknown" or not result["usage"]["complete"]:
                    break
        asyncio.run(run_all())
        summary = {"expected": len(ids), "completed": sum(r["status"] == "completed" for r in results.values()),
                   "abstained": sum(r["status"] == "abstained" for r in results.values()),
                   "unknown": sum(r["status"] == "unknown" for r in results.values()),
                   "unknown_usage": sum(not r["usage"]["complete"] for r in results.values()),
                   "missing_ids": [i for i in ids if i not in results],
                   "correct": sum(r["is_correct"] is True for r in results.values())}
        summary["accuracy_fixed_denominator"] = summary["correct"] / len(ids)
        summary["complete"] = not (summary["unknown"] or summary["unknown_usage"] or summary["missing_ids"])
        write_json(args.output_dir / "sampling-budget.json", {
            "scope": "total_prompt_plus_completion", "source_run": str(args.output_dir.resolve()),
            "budgets": {i: r["usage"]["llm_total_tokens"] for i, r in results.items()
                        if r["usage"]["complete"] and r["status"] != "unknown" and r["usage"]["llm_total_tokens"] > 0},
            "embedding_accounting": "Recorded separately; not included in generation-token matching"})
    except Exception as exc:
        summary = {"expected": len(ids), "recorded": len(results), "complete": False,
                   "status": "failed", "error_type": type(exc).__name__,
                   "missing_ids": [i for i in ids if i not in results]}
    write_json(args.output_dir / "receipt.json", summary)
    print(json.dumps(summary))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
