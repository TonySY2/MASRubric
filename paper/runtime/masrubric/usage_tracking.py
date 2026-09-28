from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Mapping

from autogen_core.models import CreateResult, ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient


USAGE_LOG_ENV = "MASRUBRIC_USAGE_LOG"
_WRITE_LOCK = threading.Lock()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _usage_log_path() -> Path | None:
    raw_path = os.getenv(USAGE_LOG_ENV, "").strip()
    if not raw_path:
        return None
    return Path(raw_path)


def _normalize_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None

    if isinstance(usage, Mapping):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    if prompt_tokens is None and total_tokens is not None:
        prompt_tokens = total_tokens
    if completion_tokens is None:
        completion_tokens = 0

    try:
        prompt_value = int(prompt_tokens or 0)
        completion_value = int(completion_tokens or 0)
        total_value = int(total_tokens or (prompt_value + completion_value))
    except Exception:
        return None

    return {
        "prompt_tokens": prompt_value,
        "completion_tokens": completion_value,
        "total_tokens": total_value,
    }


def record_usage(
    *,
    model: str | None,
    usage: Any,
    stage: str,
    source: str | None = None,
    kind: str = "chat_completion",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    usage_dict = _normalize_usage(usage)
    usage_log_path = _usage_log_path()
    if usage_dict is None or usage_log_path is None:
        return

    event = {
        "timestamp": _now_utc_iso(),
        "kind": kind,
        "stage": stage,
        "source": source,
        "model": model,
        **usage_dict,
    }
    if metadata:
        event["metadata"] = dict(metadata)

    usage_log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, ensure_ascii=False)
    with _WRITE_LOCK:
        with usage_log_path.open("a", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")


def record_chat_completion(
    completion: Any,
    *,
    model: str | None,
    stage: str,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    record_usage(
        model=getattr(completion, "model", None) or model,
        usage=getattr(completion, "usage", None),
        stage=stage,
        source=source,
        kind="chat_completion",
        metadata=metadata,
    )


def record_embedding_response(
    response: Any,
    *,
    model: str | None,
    stage: str,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    record_usage(
        model=getattr(response, "model", None) or model,
        usage=getattr(response, "usage", None),
        stage=stage,
        source=source,
        kind="embedding",
        metadata=metadata,
    )


def load_usage_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def _new_bucket() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "call_count": 0,
    }


def _bucket_to_dict(bucket: dict[str, int]) -> dict[str, int]:
    return {
        "prompt_tokens": int(bucket["prompt_tokens"]),
        "completion_tokens": int(bucket["completion_tokens"]),
        "total_tokens": int(bucket["total_tokens"]),
        "call_count": int(bucket["call_count"]),
    }


def _model_alias(model: Any) -> str:
    raw = str(model or "unknown").strip()
    if not raw:
        return "unknown"
    normalized = raw.rstrip("/\\")
    alias = Path(normalized).name or normalized
    return alias


def _needs_local_model_info(model: Any, base_url: Any) -> bool:
    model_name = str(model or "").strip()
    if not model_name:
        return False

    # OpenAI-family names can continue using AutoGen's built-in registry.
    if model_name.startswith(("gpt-", "o1", "o3", "o4")):
        return False

    if "/" in model_name or "\\" in model_name:
        return True

    endpoint = str(base_url or "").strip().lower()
    return endpoint.startswith(("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost"))


def _default_local_model_info() -> dict[str, Any]:
    return {
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.UNKNOWN,
        "structured_output": False,
    }


def summarize_usage_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    total = _new_bucket()
    by_model: dict[str, dict[str, int]] = defaultdict(_new_bucket)
    by_model_alias: dict[str, dict[str, int]] = defaultdict(_new_bucket)
    by_stage: dict[str, dict[str, int]] = defaultdict(_new_bucket)
    by_stage_and_model: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(_new_bucket))
    by_stage_and_model_alias: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(_new_bucket))

    for event in events:
        usage = _normalize_usage(event)
        if usage is None:
            continue
        model_name = str(event.get("model") or "unknown")
        model_alias = _model_alias(model_name)
        stage_name = str(event.get("stage") or "unknown")

        for bucket in (
            total,
            by_model[model_name],
            by_model_alias[model_alias],
            by_stage[stage_name],
            by_stage_and_model[stage_name][model_name],
            by_stage_and_model_alias[stage_name][model_alias],
        ):
            bucket["prompt_tokens"] += usage["prompt_tokens"]
            bucket["completion_tokens"] += usage["completion_tokens"]
            bucket["total_tokens"] += usage["total_tokens"]
            bucket["call_count"] += 1

    return {
        "event_count": total["call_count"],
        "prompt_tokens": total["prompt_tokens"],
        "completion_tokens": total["completion_tokens"],
        "total_tokens": total["total_tokens"],
        "by_model": {key: _bucket_to_dict(value) for key, value in sorted(by_model.items())},
        "by_model_alias": {key: _bucket_to_dict(value) for key, value in sorted(by_model_alias.items())},
        "by_stage": {key: _bucket_to_dict(value) for key, value in sorted(by_stage.items())},
        "by_stage_and_model": {
            stage: {model: _bucket_to_dict(bucket) for model, bucket in sorted(models.items())}
            for stage, models in sorted(by_stage_and_model.items())
        },
        "by_stage_and_model_alias": {
            stage: {model: _bucket_to_dict(bucket) for model, bucket in sorted(models.items())}
            for stage, models in sorted(by_stage_and_model_alias.items())
        },
    }


def summarize_usage_file(path: Path) -> dict[str, Any]:
    return summarize_usage_events(load_usage_events(path))


class TrackedOpenAIChatCompletionClient(OpenAIChatCompletionClient):
    def __init__(
        self,
        *,
        usage_stage: str,
        usage_source: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._usage_stage = usage_stage
        self._usage_source = usage_source
        self._usage_model = kwargs.get("model")
        if (
            "model_info" not in kwargs
            and "model_capabilities" not in kwargs
            and _needs_local_model_info(kwargs.get("model"), kwargs.get("base_url"))
        ):
            kwargs["model_info"] = _default_local_model_info()
        super().__init__(**kwargs)

    async def create(self, *args: Any, **kwargs: Any) -> CreateResult:
        result = await super().create(*args, **kwargs)
        record_usage(
            model=self._usage_model,
            usage=result.usage,
            stage=self._usage_stage,
            source=self._usage_source,
            kind="chat_completion",
        )
        return result

    async def create_stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        final_result: CreateResult | None = None
        async for chunk in super().create_stream(*args, **kwargs):
            if isinstance(chunk, CreateResult):
                final_result = chunk
            yield chunk
        if final_result is not None:
            record_usage(
                model=self._usage_model,
                usage=final_result.usage,
                stage=self._usage_stage,
                source=self._usage_source,
                kind="chat_completion",
            )
