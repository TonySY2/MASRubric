"""Persist request usage independently of agent history and pruning resets."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from AgentDropout.usage import current_usage, summarize_events, usage_scope


_sample_state = ContextVar("experiment_sample_state", default=None)


def mark_sample_failed(error):
    state = _sample_state.get()
    if state is not None:
        state.update(status="failed", error_type=type(error).__name__)


def attach_usage(result):
    """Keep legacy result fields and add the current full request ledger."""
    ledger = current_usage()
    state = _sample_state.get()
    if ledger is not None:
        result["token_usage"] = ledger.summary()
    if state is not None:
        result["run_id"] = state["run_id"]
        result["framework"] = state["framework"]
        state["result_written"] = True
    return result


def sample_id(instance):
    for key in ("id", "task_id", "question_id", "unique_id"):
        if key in instance:
            return str(instance[key])
    return "unknown"


class UsageRun:
    """One invocation, with unique sidecars so reruns never mix token costs.

    All samples, including caught sample failures, appear in the sidecars.
    Usage is recorded before team construction to include embedding setup.
    Prompts, answers, endpoint URLs and API keys are not copied to these files.
    """

    def __init__(self, output_file, framework="dynamic", configuration=None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
        self.framework = framework
        self.configuration = configuration or {}
        path = Path(output_file)
        if path.exists():
            raise FileExistsError(f"Result file already exists: {path}. Choose a new --out_file to keep runs separate.")
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = path.with_name(f"{path.stem}_{self.run_id}")
        self.events_path = Path(str(prefix) + ".usage.jsonl")
        self.samples_path = Path(str(prefix) + ".samples.jsonl")
        self.summary_path = Path(str(prefix) + ".usage.summary.json")
        self.events = []
        self.samples = []
        self._write_summary()
        print(f"[Usage] Run summary: {self.summary_path}")

    def raise_if_failed(self):
        failed = sum(sample["status"] != "completed" for sample in self.samples)
        if failed:
            raise RuntimeError(f"{failed} sample(s) failed; partial usage was saved to {self.summary_path}")

    @contextmanager
    def sample(self, task_id):
        state = {"sample_id": str(task_id), "run_id": self.run_id,
                 "framework": self.framework, "status": "completed",
                 "result_written": False}
        token = _sample_state.set(state)
        try:
            with usage_scope(str(task_id)) as ledger:
                try:
                    yield ledger
                except BaseException as error:
                    mark_sample_failed(error)
                    raise
                finally:
                    if not state["result_written"] and state["status"] == "completed":
                        state["status"] = "no_result"
                    state["token_usage"] = ledger.summary()
                    self.events.extend(ledger.events)
                    self.samples.append(dict(state))
                    with self.events_path.open("a", encoding="utf-8") as handle:
                        for event in ledger.events:
                            handle.write(json.dumps({"run_id": self.run_id, **event}, ensure_ascii=False) + "\n")
                    with self.samples_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(state, ensure_ascii=False) + "\n")
                    self._write_summary()
        finally:
            _sample_state.reset(token)

    def _write_summary(self):
        summary = {
            "schema_version": 1,
            "run_id": self.run_id,
            "framework": self.framework,
            "configuration": self.configuration,
            "samples_attempted": len(self.samples),
            "samples_completed": sum(s["status"] == "completed" for s in self.samples),
            "samples_failed": sum(s["status"] != "completed" for s in self.samples),
            "token_usage": summarize_events(self.events),
            "events_file": self.events_path.name,
            "samples_file": self.samples_path.name,
        }
        temporary = self.summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.summary_path)
