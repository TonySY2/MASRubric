from __future__ import annotations

import re
from typing import Any


ONLINE_INTERFACE_SHAPES = (
    "function_api",
    "class_api",
    "stdin_stdout",
    "file_or_env",
    "mixed",
    "unknown",
)
METRIC_INTERFACE_SHAPES = ONLINE_INTERFACE_SHAPES + ("any",)

ONLINE_IO_CONTRACTS = (
    "return_value",
    "stdout",
    "mutation",
    "exception",
    "file_io",
    "dependency",
    "unknown",
)
METRIC_IO_CONTRACTS = ONLINE_IO_CONTRACTS + ("any",)

MATCH_PROFILE_SCHEMA = {
    "interface_shape": list(ONLINE_INTERFACE_SHAPES),
    "io_contract": list(ONLINE_IO_CONTRACTS),
    "metric_interface_shape_extra": ["any"],
    "metric_io_contract_extra": ["any"],
}

_STDIN_HINTS = (
    "stdin",
    "stdout",
    "input format",
    "output format",
    "read from input",
    "print the answer",
    "read input",
    "write output",
)
_FILE_HINTS = (
    "file",
    "path",
    "environment",
    "package",
    "module",
    "import",
    "dependency",
    "os.environ",
)
_EXCEPTION_HINTS = (
    "raise",
    "exception",
    "error if",
    "should fail",
    "throws",
)
_MUTATION_HINTS = (
    "in-place",
    "in place",
    "modify the input",
    "mutate",
    "update the list",
    "change the array",
    "change the matrix",
)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _count_tokens_like_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+|[^\s]", text))


def build_metric_embedding_text(metric: dict[str, Any]) -> str:
    evaluator_prompt = metric.get("evaluator_prompt") or {}
    profile = normalize_match_profile(metric.get("match_profile"), metric_side=True)
    return "\n".join(
        [
            f"name: {_normalize_text(metric.get('name'))}",
            f"definition: {_normalize_text(metric.get('detailed_definition'))}",
            f"trigger: {_normalize_text(evaluator_prompt.get('trigger_condition'))}",
            f"risk: {_normalize_text(evaluator_prompt.get('risk_alert'))}",
            f"shape: {profile['interface_shape']}",
            f"io: {profile['io_contract']}",
            f"hint: {_normalize_text(profile['match_hint'])}",
        ]
    )


def build_query_embedding_text(profile: dict[str, Any]) -> str:
    normalized = normalize_match_profile(profile, metric_side=False)
    return "\n".join(
        [
            f"shape: {normalized['interface_shape']}",
            f"io: {normalized['io_contract']}",
            f"hint: {_normalize_text(normalized['match_hint'])}",
        ]
    )


def normalize_match_profile(profile: Any, *, metric_side: bool) -> dict[str, str]:
    allowed_shapes = METRIC_INTERFACE_SHAPES if metric_side else ONLINE_INTERFACE_SHAPES
    allowed_io = METRIC_IO_CONTRACTS if metric_side else ONLINE_IO_CONTRACTS
    default_shape = "any" if metric_side else "unknown"
    default_io = "any" if metric_side else "unknown"

    raw = profile if isinstance(profile, dict) else {}
    shape = _normalize_text(raw.get("interface_shape")).lower().replace("-", "_")
    io_contract = _normalize_text(raw.get("io_contract")).lower().replace("-", "_")
    match_hint = _normalize_text(raw.get("match_hint"))

    if shape not in allowed_shapes:
        shape = default_shape
    if io_contract not in allowed_io:
        io_contract = default_io
    if not match_hint:
        match_hint = "General code correctness check for the visible task and implementation risk."

    return {
        "interface_shape": shape,
        "io_contract": io_contract,
        "match_hint": match_hint,
    }


def validate_match_profile(profile: Any, *, metric_side: bool) -> list[str]:
    errors: list[str] = []
    normalized = normalize_match_profile(profile, metric_side=metric_side)
    allowed_shapes = set(METRIC_INTERFACE_SHAPES if metric_side else ONLINE_INTERFACE_SHAPES)
    allowed_io = set(METRIC_IO_CONTRACTS if metric_side else ONLINE_IO_CONTRACTS)

    raw = profile if isinstance(profile, dict) else {}
    raw_shape = _normalize_text(raw.get("interface_shape")).lower().replace("-", "_")
    raw_io = _normalize_text(raw.get("io_contract")).lower().replace("-", "_")
    raw_hint = _normalize_text(raw.get("match_hint"))

    if raw_shape not in allowed_shapes:
        errors.append(f"invalid interface_shape={raw_shape or '<empty>'}")
    if raw_io not in allowed_io:
        errors.append(f"invalid io_contract={raw_io or '<empty>'}")
    if not raw_hint:
        errors.append("empty match_hint")

    added_token_estimate = (
        _count_tokens_like_words(normalized["interface_shape"])
        + _count_tokens_like_words(normalized["io_contract"])
        + _count_tokens_like_words(normalized["match_hint"])
    )
    if metric_side and added_token_estimate > 150:
        errors.append(f"profile too long: approx_tokens={added_token_estimate}")
    if not metric_side and added_token_estimate > 300:
        errors.append(f"profile too long: approx_tokens={added_token_estimate}")
    return errors


def has_profile_conflict(metric_profile: Any, online_profile: Any) -> bool:
    metric = normalize_match_profile(metric_profile, metric_side=True)
    online = normalize_match_profile(online_profile, metric_side=False)

    metric_shape = metric["interface_shape"]
    online_shape = online["interface_shape"]
    if (
        metric_shape not in {"any", "unknown", "mixed"}
        and online_shape not in {"unknown", "mixed"}
        and metric_shape != online_shape
    ):
        return True

    metric_io = metric["io_contract"]
    online_io = online["io_contract"]
    if metric_io not in {"any", "unknown"} and online_io != "unknown" and metric_io != online_io:
        return True

    return False


def infer_profile_heuristically(task: str, agent_output: str) -> dict[str, str]:
    task_text = _normalize_text(task)
    output_text = _normalize_text(agent_output)
    merged = f"{task_text}\n{output_text}".lower()

    has_stdin = any(token in merged for token in _STDIN_HINTS)
    has_file_env = any(token in merged for token in _FILE_HINTS)
    has_class = "class solution" in merged or re.search(r"\bclass\s+[a-z_][a-z0-9_]*\b", merged) is not None
    has_function = re.search(r"\bdef\s+[a-z_][a-z0-9_]*\s*\(", merged) is not None

    if has_stdin and (has_class or has_function):
        interface_shape = "mixed"
    elif has_stdin:
        interface_shape = "stdin_stdout"
    elif has_file_env:
        interface_shape = "file_or_env"
    elif has_class:
        interface_shape = "class_api"
    elif has_function or "starter code" in merged:
        interface_shape = "function_api"
    else:
        interface_shape = "unknown"

    if has_file_env:
        io_contract = "dependency" if any(token in merged for token in ("package", "module", "import", "dependency")) else "file_io"
    elif has_stdin:
        io_contract = "stdout"
    elif any(token in merged for token in _MUTATION_HINTS):
        io_contract = "mutation"
    elif any(token in merged for token in _EXCEPTION_HINTS):
        io_contract = "exception"
    elif interface_shape in {"function_api", "class_api"}:
        io_contract = "return_value"
    else:
        io_contract = "unknown"

    risk_bits = []
    if interface_shape == "stdin_stdout":
        risk_bits.append("stdin parsing and stdout formatting")
    if interface_shape == "class_api":
        risk_bits.append("class method signature and return behavior")
    if interface_shape == "function_api":
        risk_bits.append("function signature and returned result")
    if io_contract == "mutation":
        risk_bits.append("in-place state updates")
    if io_contract == "exception":
        risk_bits.append("required exception behavior")
    if io_contract in {"file_io", "dependency"}:
        risk_bits.append("environment and dependency assumptions")
    if not risk_bits:
        risk_bits.append("core implementation logic")

    task_excerpt = task_text[:160]
    output_excerpt = output_text[:200]
    match_hint = (
        f"Task context: {task_excerpt}. "
        f"Visible agent evidence: {output_excerpt}. "
        f"Prioritize {', '.join(risk_bits)}."
    )
    return {
        "interface_shape": interface_shape,
        "io_contract": io_contract,
        "match_hint": _normalize_text(match_hint),
    }
