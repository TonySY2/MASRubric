from __future__ import annotations

import builtins
import contextvars
import itertools
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


TRACE_FILE_ENV = "AGENTDROPOUT_ATOMIC_TRACE_FILE"
RUN_ID_ENV = "AGENTDROPOUT_ATOMIC_TRACE_RUN_ID"
TASK_ENV = "AGENTDROPOUT_ATOMIC_TRACE_TASK"
EXPERIMENT_ENV = "AGENTDROPOUT_ATOMIC_TRACE_EXPERIMENT"

_installed = False
_true_print = builtins.print
_write_lock = threading.Lock()
_global_seq = itertools.count(1)
_record_seq: dict[str, int] = {}
_current_record_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "atomic_trace_record_id", default=None
)
_pending_events: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "atomic_trace_pending_events", default=None
)

_START_PATTERNS = [
    re.compile(r"^\s*开始处理:\s*(?P<id>\S+)\s*$"),
    re.compile(r"Processing Task\s+(?P<id>\S+)"),
]
_END_PATTERNS = [
    re.compile(r"^\s*完成处理:\s*(?P<id>\S+)"),
    re.compile(r"Task\s+(?P<id>\S+).*?(CRITICAL ERROR|发生错误|error)", re.IGNORECASE),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_default(value: Any) -> str:
    return repr(value)


def _render_print_payload(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    if sep is None:
        sep = " "
    if end is None:
        end = "\n"
    return str(sep).join(str(arg) for arg in args) + str(end)


def _infer_record_id(text: str) -> str | None:
    first_line = text.splitlines()[0] if text.splitlines() else text
    for pattern in _START_PATTERNS:
        match = pattern.search(first_line)
        if match:
            return match.group("id")
    for pattern in _END_PATTERNS:
        match = pattern.search(first_line)
        if match:
            return match.group("id")
    return None


def _event_type(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "blank"
    if any(pattern.search(stripped) for pattern in _START_PATTERNS):
        return "task_start"
    if stripped.startswith("完成处理:"):
        return "task_end"
    if "CRITICAL ERROR" in stripped or "发生错误" in stripped:
        return "exception"
    if stripped.startswith(">>> [Phase"):
        return "phase"
    if "Generated Content" in stripped:
        return "agent_message_header"
    if "Auditing" in stripped or "审计轮次" in stripped:
        return "audit_header"
    if "Final Decision" in stripped or "最终" in stripped:
        return "final_decision"
    return "print"


def _next_record_seq(record_id: str | None) -> int | None:
    if record_id is None:
        return None
    with _write_lock:
        next_value = _record_seq.get(record_id, 0) + 1
        _record_seq[record_id] = next_value
        return next_value


def _write_events(trace_file: Path, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with trace_file.open("a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")


def _make_event(
    *,
    text: str,
    record_id: str | None,
    destination: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": os.environ.get(RUN_ID_ENV) or None,
        "experiment": os.environ.get(EXPERIMENT_ENV) or None,
        "task": os.environ.get(TASK_ENV) or None,
        "record_id": record_id,
        "global_seq": next(_global_seq),
        "record_seq": _next_record_seq(record_id),
        "timestamp": _utc_now(),
        "event_type": _event_type(text),
        "destination": destination,
        "content": text,
    }


def _destination_name(file_obj: Any) -> str:
    if file_obj is None or file_obj is sys.stdout:
        return "stdout"
    if file_obj is sys.stderr:
        return "stderr"
    name = getattr(file_obj, "name", None)
    if isinstance(name, str):
        return name
    return type(file_obj).__name__


def install_from_env() -> bool:
    global _installed
    if _installed:
        return True
    trace_path = os.environ.get(TRACE_FILE_ENV)
    if not trace_path:
        return False

    trace_file = Path(trace_path)
    original_print = builtins.print

    def atomic_print(*args: Any, **kwargs: Any) -> None:
        text = _render_print_payload(args, kwargs)
        file_obj = kwargs.get("file", None)
        destination = _destination_name(file_obj)

        record_id = _current_record_id.get()
        inferred_id = _infer_record_id(text)
        if inferred_id is not None:
            record_id = inferred_id
            _current_record_id.set(record_id)

        events_to_write: list[dict[str, Any]] = []
        if record_id is not None:
            pending = _pending_events.get()
            if pending:
                for pending_event in pending:
                    pending_event["record_id"] = record_id
                    pending_event["record_seq"] = _next_record_seq(record_id)
                events_to_write.extend(pending)
                _pending_events.set(None)
            events_to_write.append(
                _make_event(text=text, record_id=record_id, destination=destination)
            )
        else:
            event = _make_event(text=text, record_id=None, destination=destination)
            if text.startswith("--- [") and "Processing Task" not in text:
                pending = _pending_events.get()
                if pending is None:
                    pending = []
                    _pending_events.set(pending)
                pending.append(event)
            else:
                events_to_write.append(event)

        try:
            _write_events(trace_file, events_to_write)
        except Exception as exc:
            original_print(
                f"[atomic_trace] failed to write trace event: {exc}",
                file=sys.stderr,
            )

        original_print(*args, **kwargs)

    builtins.print = atomic_print
    _installed = True
    return True


def _natural_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def load_events(trace_file: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with trace_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append(
                    {
                        "record_id": None,
                        "global_seq": len(events) + 1,
                        "record_seq": None,
                        "timestamp": None,
                        "event_type": "bad_json",
                        "content": line,
                    }
                )
    return events


def render_events(events: list[dict[str, Any]], out: TextIO) -> dict[str, Any]:
    global_events: list[dict[str, Any]] = []
    by_record: dict[str, list[dict[str, Any]]] = {}

    for event in events:
        record_id = event.get("record_id")
        if record_id is None or record_id == "":
            global_events.append(event)
        else:
            by_record.setdefault(str(record_id), []).append(event)

    out.write("=== Atomic Trace Rendered Log ===\n")
    out.write(f"Total Events: {len(events)}\n")
    out.write(f"Total Records: {len(by_record)}\n\n")

    if global_events:
        out.write("========================================\n")
        out.write("=== GLOBAL EVENTS ===\n")
        out.write("========================================\n")
        for event in sorted(global_events, key=lambda item: int(item.get("global_seq") or 0)):
            out.write(str(event.get("content") or ""))
            if not str(event.get("content") or "").endswith("\n"):
                out.write("\n")
        out.write("\n")

    for record_id in sorted(by_record, key=_natural_key):
        out.write("\n========================================\n")
        out.write(f"=== TASK ID: {record_id} ===\n")
        out.write("========================================\n")
        record_events = sorted(
            by_record[record_id],
            key=lambda item: (
                int(item.get("record_seq") or 0),
                int(item.get("global_seq") or 0),
            ),
        )
        for event in record_events:
            content = str(event.get("content") or "")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")

    return {
        "events": len(events),
        "global_events": len(global_events),
        "records": len(by_record),
        "records_with_events": {
            record_id: len(items) for record_id, items in sorted(by_record.items(), key=lambda kv: _natural_key(kv[0]))
        },
    }


def render_trace_file(trace_file: Path, output_file: Path) -> dict[str, Any]:
    events = load_events(trace_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out:
        summary = render_events(events, out)
    return summary
