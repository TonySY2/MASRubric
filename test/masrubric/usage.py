"""Per-call usage accounting, following the reference experiment stage names.

Only provider-returned usage is counted. Missing usage and failed requests remain
visible as unknown values; SDK-internal transport retries cannot be reconstructed
when the provider returns no usage for them. Embedding usage is kept separate.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Iterator, Mapping


_ACTIVE_LEDGER: ContextVar[UsageLedger | None] = ContextVar("masrubric_usage", default=None)
_USAGE_PHASE: ContextVar[str] = ContextVar("masrubric_usage_phase", default="primary")
RECTIFIER_STAGES = frozenset({
    "supervisor_summary", "supervisor_rerank", "supervisor_audit", "supervisor_audit_batch",
})
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if number < 0 or (not isinstance(value, str) and number != value):
        return None
    return number


def _usage_values(response: Any, kind: str) -> dict[str, Any]:
    # Accept an SDK response, an AutoGen CreateResult, or a usage object itself.
    usage = _get(response, "usage", response)
    prompt = _token_count(_get(usage, "prompt_tokens", _get(usage, "input_tokens")))
    completion = _token_count(_get(usage, "completion_tokens", _get(usage, "output_tokens")))
    total = _token_count(_get(usage, "total_tokens"))
    if kind == "embedding":
        # Embedding requests have no generated output. This is a semantic zero,
        # while their unknown input/total usage must still remain unknown.
        completion = 0
        if prompt is None and total is not None:
            prompt = total
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    values = dict(zip(TOKEN_FIELDS, (prompt, completion, total)))
    known = all(value is not None for value in values.values())
    consistent = not known or prompt + completion == total
    if known and consistent:
        status = "complete"
    elif known:
        status = "inconsistent"
    elif prompt is None and total is None and (completion is None or kind == "embedding"):
        status = "missing"
    else:
        status = "partial"
    return {**values, "usage_status": status}


def _model_label(model: Any) -> str | None:
    if model is None:
        return None
    label = str(model)
    # Do not serialize local model directories or endpoint URLs into traces.
    if "://" in label:
        return "custom-model"
    if label.startswith(("/", "\\")) or (len(label) > 1 and label[1] == ":"):
        return label.rstrip("/\\").replace("\\", "/").rsplit("/", 1)[-1]
    return label


def _bucket(events: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "call_count": len(events),
        "failed_calls": sum(event.get("outcome") == "failed" for event in events),
        "missing_usage_calls": sum(event.get("usage_status") != "complete" for event in events),
        "complete": all(event.get("usage_status") == "complete" for event in events),
    }
    for key in TOKEN_FIELDS:
        observed = sum(event[key] for event in events if event.get(key) is not None)
        result["observed_" + key] = observed
        result[key] = observed if all(event.get(key) is not None for event in events) else None
    return result


def summarize_events(events: list[dict[str, Any]], sample_id: str | None = None) -> dict[str, Any]:
    """Summarize provider usage, with strict totals and partial observed sums."""
    stages = sorted({str(event["stage"]) for event in events})
    phases = sorted({str(event["phase"]) for event in events})
    result: dict[str, Any] = {
        "schema_version": 1,
        "measurement": "provider_response_usage",
        "call_count": len(events),
        "failed_calls": sum(event.get("outcome") == "failed" for event in events),
        "missing_usage_calls": sum(event.get("usage_status") != "complete" for event in events),
        "llm": _bucket([event for event in events if event["kind"] == "llm"]),
        "embedding": _bucket([event for event in events if event["kind"] == "embedding"]),
        "online_rectifier_llm": _bucket([
            event for event in events if event["kind"] == "llm" and event["stage"] in RECTIFIER_STAGES
        ]),
        "by_stage": {stage: _bucket([event for event in events if event["stage"] == stage]) for stage in stages},
        "by_phase": {
            phase: {
                kind: _bucket([event for event in events if event["phase"] == phase and event["kind"] == kind])
                for kind in ("llm", "embedding")
            } for phase in phases
        },
    }
    if sample_id is not None:
        result["sample_id"] = str(sample_id)
    return result


summarize_usage_events = summarize_events


@dataclass
class UsageLedger:
    sample_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    phase: str = "primary"

    def summary(self) -> dict[str, Any]:
        return summarize_events(self.events, self.sample_id)


@contextmanager
def usage_scope(sample_id: str) -> Iterator[UsageLedger]:
    """Keep every call for one sample, including calls before a fallback reset."""
    ledger = UsageLedger(str(sample_id))
    ledger_token = _ACTIVE_LEDGER.set(ledger)
    phase_token = _USAGE_PHASE.set("primary")
    try:
        yield ledger
    finally:
        _USAGE_PHASE.reset(phase_token)
        _ACTIVE_LEDGER.reset(ledger_token)


def current_usage() -> UsageLedger | None:
    return _ACTIVE_LEDGER.get()


def set_usage_phase(phase: str) -> None:
    _USAGE_PHASE.set(str(phase))
    ledger = current_usage()
    if ledger is not None:
        # AutoGen actors may retain the ContextVar context from their first run.
        # The sample-owned ledger lets those actors see a later fallback phase.
        ledger.phase = str(phase)


def record_usage(
    stage: str,
    response: Any,
    *,
    kind: str = "llm",
    model: str | None = None,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    failed: bool = False,
    error_type: str | None = None,
) -> None:
    ledger = current_usage()
    if ledger is None:
        return
    if kind not in {"llm", "embedding"}:
        raise ValueError("Usage kind must be llm or embedding.")
    event = {
        "call_index": len(ledger.events) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_id": ledger.sample_id,
        "phase": ledger.phase,
        "stage": stage,
        "kind": kind,
        "source": source,
        "model": _model_label(_get(response, "model") or model),
        "outcome": "failed" if failed else "success",
        **_usage_values(response, kind),
    }
    if error_type:
        # Exception text may contain endpoint URLs or request bodies.
        event["error_type"] = error_type
    if metadata:
        event["metadata"] = {
            key: value for key, value in metadata.items()
            if key in {"attempt", "agent_attempt", "audit_attempt", "metric", "role"}
            and isinstance(value, (str, int, float, bool, type(None)))
        }
    ledger.events.append(event)


async def tracked_create(
    create_callable: Callable[..., Any],
    *,
    stage: str,
    source: str | None = None,
    kind: str = "llm",
    metadata: Mapping[str, Any] | None = None,
    **request_kwargs: Any,
) -> Any:
    """Record immediately after a response, before downstream parsing can fail."""
    try:
        response = await create_callable(**request_kwargs)
    except (Exception, asyncio.CancelledError) as error:
        record_usage(stage, None, kind=kind, model=request_kwargs.get("model"), source=source,
                     metadata=metadata, failed=True, error_type=type(error).__name__)
        raise
    record_usage(stage, response, kind=kind, model=request_kwargs.get("model"), source=source, metadata=metadata)
    return response


def tracked_embedding_create(
    create_callable: Callable[..., Any],
    *,
    stage: str,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    **request_kwargs: Any,
) -> Any:
    try:
        response = create_callable(**request_kwargs)
    except Exception as error:
        record_usage(stage, None, kind="embedding", model=request_kwargs.get("model"), source=source,
                     metadata=metadata, failed=True, error_type=type(error).__name__)
        raise
    record_usage(stage, response, kind="embedding", model=request_kwargs.get("model"), source=source, metadata=metadata)
    return response


def safe_request_usage(response: Any) -> Any:
    """Provide optional AutoGen message metadata without inventing zero usage."""
    usage = _usage_values(response, "llm")
    if usage["prompt_tokens"] is None or usage["completion_tokens"] is None:
        return None
    from autogen_core.models import RequestUsage
    return RequestUsage(prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"])


# The accounting helpers remain importable without inference dependencies. The
# selector subclass is resolved lazily when the team factory requests it.
_SELECTOR_CALL: ContextVar[dict[str, Any] | None] = ContextVar("selector_usage_call", default=None)


def __getattr__(name: str) -> Any:
    if name != "TrackedOpenAIChatCompletionClient":
        raise AttributeError(name)
    from autogen_core.models import ModelFamily
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_ext.models.openai import _model_info

    class ObservedStream:
        def __init__(self, stream: Any, state: dict[str, Any]) -> None:
            self._stream = stream
            self._state = state

        def __getattr__(self, key: str) -> Any:
            return getattr(self._stream, key)

        async def __aiter__(self) -> AsyncGenerator[Any, None]:
            async for chunk in self._stream:
                if _get(chunk, "usage") is not None:
                    self._state["response"] = chunk
                yield chunk

    class TrackedOpenAIChatCompletionClient(OpenAIChatCompletionClient):
        def __init__(self, *, usage_stage: str = "selector", usage_source: str | None = None, **kwargs: Any) -> None:
            self._usage_stage = usage_stage
            self._usage_source = usage_source
            self._usage_model = kwargs.get("model")
            if (
                kwargs.get("model_info") is None
                and kwargs.get("model_capabilities") is None
                and isinstance(self._usage_model, str)
                and self._usage_model
            ):
                try:
                    _model_info.get_info(self._usage_model)
                except ValueError:
                    # The reference runs use OpenAI-compatible text servers
                    # whose served model names are outside AutoGen's registry.
                    kwargs["model_info"] = {
                        "vision": False,
                        "function_calling": False,
                        "json_output": False,
                        "family": ModelFamily.UNKNOWN,
                        "structured_output": False,
                    }
            super().__init__(**kwargs)
            # AutoGen 0.7.5 substitutes zero when raw provider usage is absent.
            # Observe the pinned SDK's raw response before that conversion.
            resource = self._client.chat.completions
            resource.create = self._observe_provider(resource.create)
            beta = getattr(self._client, "beta", None)
            if beta is not None:
                parsed_resource = beta.chat.completions
                parsed_resource.parse = self._observe_provider(parsed_resource.parse)

        def _record_state(self, state: dict[str, Any], *, failed: bool = False, error_type: str | None = None) -> None:
            if state["recorded"]:
                return
            record_usage(self._usage_stage, state.get("response"), model=self._usage_model,
                         source=self._usage_source, failed=failed, error_type=error_type)
            state["recorded"] = True

        def _observe_provider(self, create_callable: Callable[..., Any]) -> Callable[..., Any]:
            async def observed(*args: Any, **kwargs: Any) -> Any:
                state = _SELECTOR_CALL.get()
                try:
                    response = await create_callable(*args, **kwargs)
                except (Exception, asyncio.CancelledError) as error:
                    if state is not None:
                        self._record_state(state, failed=True, error_type=type(error).__name__)
                    raise
                if state is not None:
                    if kwargs.get("stream"):
                        return ObservedStream(response, state)
                    state["response"] = response
                    self._record_state(state)
                return response
            return observed

        async def create(self, *args: Any, **kwargs: Any) -> Any:
            state: dict[str, Any] = {"recorded": False}
            token = _SELECTOR_CALL.set(state)
            try:
                result = await super().create(*args, **kwargs)
                # A future SDK code path without a raw-response hook remains
                # explicitly unknown instead of trusting synthetic zero usage.
                self._record_state(state)
                return result
            except (Exception, asyncio.CancelledError) as error:
                self._record_state(state, failed=True, error_type=type(error).__name__)
                raise
            finally:
                _SELECTOR_CALL.reset(token)

        async def create_stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            state: dict[str, Any] = {"recorded": False}
            token = _SELECTOR_CALL.set(state)
            finished = False
            try:
                async for chunk in super().create_stream(*args, **kwargs):
                    yield chunk
                finished = True
                self._record_state(state)
            except (Exception, asyncio.CancelledError) as error:
                self._record_state(state, failed=True, error_type=type(error).__name__)
                raise
            finally:
                if not finished and not state["recorded"]:
                    self._record_state(state, failed=True, error_type="StreamClosed")
                _SELECTOR_CALL.reset(token)

    globals()[name] = TrackedOpenAIChatCompletionClient
    return TrackedOpenAIChatCompletionClient
